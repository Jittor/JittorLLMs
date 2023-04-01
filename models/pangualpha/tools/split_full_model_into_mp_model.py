import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch

from megatron import mpu
from megatron.checkpointing import ensure_directory_exists
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import rebuild_tokenizer
from megatron.global_vars import _parse_args


def get_model(model_type):

    if model_type == 'BERT':
        from pretrain_bert import model_provider
    elif model_type == 'GPT2':
        from pretrain_gpt2 import model_provider
    elif model_type == 'Pangu':
        from pretrain_gpt2 import model_provider
    elif model_type == 'RACE':
        from tasks.race.finetune import model_provider
    elif model_type == ['MNLI', 'QQP']:
        num_classes = 2
        if model_type == 'MNLI':
            num_classes = 3
        from megatron.model.classification import Classification
        def model_provider():
            return Classification(num_classes=num_classes, num_tokentypes=2)
    else:
        raise Exception('unrecognized model type: {}'.format(model_type))

    model = model_provider()
    model = model.half()

    return model


def get_mp_split_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp merge')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['BERT', 'GPT2', 'Pangu', 'RACE', 'MNLI', 'QQP'],
                       help='Type of the mdoel.')
    group.add_argument('--num-mp-model', type=int, required=True,
                       help='Number of mp model you want to divid into.')
    group.add_argument('--mp-model-save', type=str, required=True,
                       help='Dir of saving model parallel model.')

    return parser


def get_full_model_loadCkpt(args):
    args.model_parallel_size = 1
    tokenizer = rebuild_tokenizer(args)

    mpu.initialize.set_model_parallel_world_size(1)
    mpu.initialize.set_model_parallel_rank(0)
    model = get_model(args.model_type)


    tracker_filename = get_checkpoint_tracker_filename(args.load)
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)
    checkpoint_name = get_checkpoint_name(args.load, iteration)
    sd = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(sd['model'])

    return model


def get_one_partition_model(args, rank):
    args.model_parallel_size = args.num_mp_model
    tokenizer = rebuild_tokenizer(args)

    mpu.initialize.set_model_parallel_world_size(args.num_mp_model)
    mpu.initialize.set_model_parallel_rank(rank)
    model = get_model(args.model_type)

    return model


def split_into_partitions(tensor, num_partitions, partition_dim, stride, rank):

    per_partition_size = mpu.utils.divide(tensor.size(partition_dim),
                                          num_partitions)
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size, stride)

    partitions_list = torch.split(tensor,
                                  per_partition_per_stride_size,
                                  dim=partition_dim)

    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(partitions_list[i::num_partitions],
                              dim=partition_dim)
        partitions.append(partition)

    return partitions[rank]


def split_full_model(full_parameter, partition_parameter, partition_dim, stride,
                     rank, num_partitions):

    per_partition_size = partition_parameter.size(partition_dim)

    def split():
        with torch.no_grad():
            if (per_partition_size * num_partitions) == full_parameter.size(
                    partition_dim):
                partition_parameter.data.copy_(
                    split_into_partitions(
                        full_parameter, num_partitions, partition_dim, stride, rank
                    ).data
                )
            else:
                assert full_parameter.size(partition_dim) \
                       <= per_partition_size * num_partitions

                dim_diff = (per_partition_size * num_partitions) - \
                           full_parameter.size(partition_dim)
                print('     ***WARNING*** sizes do not match. Will add '
                      'the merged partitions by {} along dimension {} '
                      'to increase the size from {} to {} ...'.format(
                    dim_diff,
                    partition_dim,
                    full_parameter.size(partition_dim),
                    per_partition_size * num_partitions)
                )

                size = list(full_parameter.size())
                size[partition_dim] = dim_diff
                add_tensor = torch.ones(size)
                full_parameter_ = torch.cat((full_parameter.data, add_tensor),
                                            partition_dim)

                assert full_parameter_.size(partition_dim) \
                       == per_partition_size * num_partitions

                partition_parameter.data.copy_(
                    split_into_partitions(
                        full_parameter_, num_partitions, partition_dim, stride, rank
                    ).data
                )

    # If stride is 1, then do simple concatination.
    if stride == 1:
        split()
        return
    else :
        raise ValueError("stride must be 1")
    return


def load_one_partition_model(partition_model, full_model, rank):

    full_params_gen = full_model.named_parameters()
    partition_params_gen = partition_model.named_parameters()
    while True:
        try:

            # Get the params and check names.
            name, full_param = next(full_params_gen)
            partition_name, partition_param = next(partition_params_gen)

            print(' > working on {} ...'.format(name))
            print('     merged         type: {}, size: {}'.format(
                full_param.dtype, list(full_param.size())))
            assert partition_name == name
            print('     partition {}    type: {}, size: {}'.format(
                rank, partition_param.dtype, list(partition_param.size())))

            # For the non-parallel parameters, simply copy from full model.
            if not hasattr(full_param, 'model_parallel'):
                print('     none-parallel parameter, simple copy from full model')
                with torch.no_grad():
                    partition_param.data.copy_(full_param.data)
            # For parallel parameters, split full model
            else:
                print('     split full model with stride {} along '
                      'dimention {}'.format(full_param.stride,
                                            full_param.partition_dim))

                split_full_model(full_param, partition_param,
                                 full_param.partition_dim,
                                 full_param.stride,
                                 rank,
                                 args.num_mp_model)

                # split_to_partitions(full_param,
                #                  partitions_param,
                #                  full_param.partition_dim,
                #                  full_param.stride)

        except StopIteration:
            break


if __name__ == '__main__':

    args = _parse_args(extra_args_provider=get_mp_split_args)
    save = args.mp_model_save

    iteration = 'iter_0007800'  # inter_***    any id is ok
    modelName = 'model_optim_rng.pt'
    full_model = get_full_model_loadCkpt(args)
    for rank in range(args.num_mp_model):
        partition_model = get_one_partition_model(args, rank)
        load_one_partition_model(partition_model, full_model,
                                 rank)

        sd = {}
        sd['model'] = partition_model.state_dict_for_save_checkpoint()
        sd['iteration'] = iteration
        rankName = 'mp_rank_{:02d}'.format(rank)
        savePath='/'.join([save,iteration,rankName,modelName])
        ensure_directory_exists(savePath)
        torch.save(sd, savePath)

    recordFile = open(get_checkpoint_tracker_filename(save), 'w')
    recordFile.writelines(iteration.split('_')[1])
    recordFile.close()
    pass

