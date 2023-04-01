# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge model parallel partitions."""

import os
import sys
from megatron.initialize import initialize_megatron

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import numpy as np

import torch

from megatron import mpu
from megatron.checkpointing import ensure_directory_exists
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import rebuild_tokenizer
from megatron.global_vars import _parse_args

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.system("export CUDA_VISIBLE_DEVICES=4,5,6,7")
#########################
# --model-type GPT2 --model-parallel-size 2 --tokenizer-type GPT2BPETokenizer --vocab-file /ghome/yands/pclproject/gpt/Megatron-LM-1.1/megatron/tokenizer/bpe_4w_pcl/vocab --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 1024 --max-position-embeddings 1024 --load /ghome/yands/model/checkPoints/megatron-1.1-pangu
# --model-type GPT2 --model-parallel-size 4 --tokenizer-type GPT2BPETokenizer --vocab-file /ghome/yands/pclproject/gpt/Megatron-LM-1.1/megatron/tokenizer/bpe_4w_pcl/vocab --num-layers 31 --hidden-size 2560 --num-attention-heads 32 --seq-length 1024 --max-position-embeddings 1024 --load /ghome/yands/model/checkPoints/megatron-1.1-pangu-2.6B
#########################


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


def get_change_ckpt_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp merge')

    group.add_argument('--model-type', type=str, required=True,
                       choices=['BERT', 'GPT2', 'Pangu', 'RACE', 'MNLI', 'QQP'],
                       help='Type of the mdoel.')
    group.add_argument('--npy-ckpt-path', type=str, required=True,
                       help='path of npy checkpoint.')

    return parser



def loadModelFromNp(sd, args):
    num_layers = args.num_layers
    npCkptPath = args.npy_ckpt_path
    languageModel = sd['model']['language_model']
    loadEmbeddingFromNp(npCkptPath, languageModel)
    transformer = sd['model']['language_model']['transformer']
    for layerID in range(num_layers):
        loadAttentionLayerFromNp(npCkptPath, transformer, layerID)
    loadQueryLayerFromNp(npCkptPath, transformer)

    transformer['final_layernorm.weight'][:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.layernorm.gamma.npy')
        ).float()
    transformer['final_layernorm.bias'][:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.layernorm.beta.npy')
        ).float()


def loadEmbeddingFromNp(npCkptPath, languageModel, vocabSize=40000):
    word_embedding_np = \
        np.load(npCkptPath+'backbone.word_embedding.embedding_table.npy')
    languageModel['embedding']['word_embeddings']['weight'][:vocabSize,:] = \
        torch.tensor(word_embedding_np).float()

    position_embeddings_np = \
        np.load(npCkptPath+'backbone.position_embedding.embedding_table.npy')
    languageModel['embedding']['position_embeddings']['weight'][:,:] = \
        torch.tensor(position_embeddings_np).float()

    topQueryEmbedding_np = \
        np.load(npCkptPath+'backbone.top_query_embedding.embedding_table.npy')
    languageModel['topQueryEmbedding']['top_query_embeddings']['weight'][:,:] = \
        torch.tensor(topQueryEmbedding_np).float()


def loadAttentionLayerFromNp(npCkptPath, transformer, layerID):
    attention_dense1_weight_np = \
        np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.dense1.weight.npy')
    attention_dense2_weight_np = \
        np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.dense2.weight.npy')
    attention_dense3_weight_np = \
        np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.dense3.weight.npy')

    attention_dense1_bias_np = \
        np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.dense1.bias.npy')
    attention_dense2_bias_np = \
        np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.dense2.bias.npy')
    attention_dense3_bias_np = \
        np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.dense3.bias.npy')


    query_weight = transformer[f'layers.{layerID}.attention.query.weight']
    key_weight = transformer[f'layers.{layerID}.attention.key.weight']
    value_weight = transformer[f'layers.{layerID}.attention.value.weight']

    query_weight[:] = torch.tensor(attention_dense1_weight_np).float()
    key_weight[:] = torch.tensor(attention_dense2_weight_np).float()
    value_weight[:] = torch.tensor(attention_dense3_weight_np).float()

    query_bias = transformer[f'layers.{layerID}.attention.query.bias']
    key_bias = transformer[f'layers.{layerID}.attention.key.bias']
    value_bias = transformer[f'layers.{layerID}.attention.value.bias']

    query_bias[:] = torch.tensor(attention_dense1_bias_np).float()
    key_bias[:] = torch.tensor(attention_dense2_bias_np).float()
    value_bias[:] = torch.tensor(attention_dense3_bias_np).float()


    att_dense_weight = transformer[f'layers.{layerID}.attention.dense.weight']
    att_dense_weight[:,:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.projection.weight.npy').transpose()
        ).float()
    att_dense_bias = transformer[f'layers.{layerID}.attention.dense.bias']
    att_dense_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.attention.projection.bias.npy')
        ).float()

    mlp_dense_h_to_4h_weight = transformer[f'layers.{layerID}.mlp.dense_h_to_4h.weight']
    mlp_dense_h_to_4h_weight[:,:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.output.mapping.weight.npy').transpose()
        ).float()
    mlp_dense_h_to_4h_bias = transformer[f'layers.{layerID}.mlp.dense_h_to_4h.bias']
    mlp_dense_h_to_4h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.output.mapping.bias.npy')
        ).float()

    mlp_dense_4h_to_h_weight = transformer[f'layers.{layerID}.mlp.dense_4h_to_h.weight']
    mlp_dense_4h_to_h_weight[:,:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.output.projection.weight.npy').transpose()
        ).float()
    mlp_dense_4h_to_h_bias = transformer[f'layers.{layerID}.mlp.dense_4h_to_h.bias']
    mlp_dense_4h_to_h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.output.projection.bias.npy')
        ).float()

    input_layernorm_weight = transformer[f'layers.{layerID}.input_layernorm.weight']
    input_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm1.gamma.npy')
        ).float()
    input_layernorm_bias = transformer[f'layers.{layerID}.input_layernorm.bias']
    input_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm1.beta.npy')
        ).float()

    post_attention_layernorm_weight = transformer[f'layers.{layerID}.post_attention_layernorm.weight']
    post_attention_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm2.gamma.npy')
        ).float()
    post_attention_layernorm_bias = transformer[f'layers.{layerID}.post_attention_layernorm.bias']
    post_attention_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm2.beta.npy')
        ).float()


    input_layernorm_weight = transformer[f'layers.{layerID}.input_layernorm.weight']
    input_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm1.gamma.npy')
        ).float()
    input_layernorm_bias = transformer[f'layers.{layerID}.input_layernorm.bias']
    input_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm1.beta.npy')
        ).float()

    post_attention_layernorm_weight = transformer[f'layers.{layerID}.post_attention_layernorm.weight']
    post_attention_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm2.gamma.npy')
        ).float()
    post_attention_layernorm_bias = transformer[f'layers.{layerID}.post_attention_layernorm.bias']
    post_attention_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.blocks.{layerID}.layernorm2.beta.npy')
        ).float()


def loadQueryLayerFromNp(npCkptPath, transformer):
    attention_dense1_weight_np = \
        np.load(npCkptPath+f'backbone.top_query_layer.attention.dense1.weight.npy')
    attention_dense1_bias_np = \
        np.load(npCkptPath+f'backbone.top_query_layer.attention.dense1.bias.npy')
    attention_dense2_weight_np = \
        np.load(npCkptPath+f'backbone.top_query_layer.attention.dense2.weight.npy')
    attention_dense2_bias_np = \
        np.load(npCkptPath+f'backbone.top_query_layer.attention.dense2.bias.npy')
    attention_dense3_weight_np = \
        np.load(npCkptPath+f'backbone.top_query_layer.attention.dense3.weight.npy')
    attention_dense3_bias_np = \
        np.load(npCkptPath+f'backbone.top_query_layer.attention.dense3.bias.npy')


    query_weight = transformer[f'topQueryLayer.attention.query.weight']
    query_weight[:, :] = \
        torch.tensor(attention_dense1_weight_np).float()
    query_bias = transformer[f'topQueryLayer.attention.query.bias']
    query_bias[:] = torch.tensor(attention_dense1_bias_np).float()

    key_weight = transformer[f'topQueryLayer.attention.key.weight']
    key_weight[:, :] = \
        torch.tensor(attention_dense2_weight_np).float()
    key_bias = transformer[f'topQueryLayer.attention.key.bias']
    key_bias[:] = torch.tensor(attention_dense2_bias_np).float()

    value_weight = transformer[f'topQueryLayer.attention.value.weight']
    value_weight[:, :] = \
        torch.tensor(attention_dense3_weight_np).float()
    value_bias = transformer[f'topQueryLayer.attention.value.bias']
    value_bias[:] = torch.tensor(attention_dense3_bias_np).float()

    att_dense_weight = transformer[f'topQueryLayer.attention.dense.weight']
    att_dense_weight[:,:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.attention.projection.weight.npy')
                .transpose()
        ).float()
    att_dense_bias = transformer[f'topQueryLayer.attention.dense.bias']
    att_dense_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.attention.projection.bias.npy')
        ).float()

    mlp_dense_h_to_4h_weight = transformer[f'topQueryLayer.mlp.dense_h_to_4h.weight']
    mlp_dense_h_to_4h_weight[:,:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.output.mapping.weight.npy')
                .transpose()
        ).float()
    mlp_dense_h_to_4h_bias = transformer[f'topQueryLayer.mlp.dense_h_to_4h.bias']
    mlp_dense_h_to_4h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.output.mapping.bias.npy')
        ).float()

    mlp_dense_4h_to_h_weight = transformer[f'topQueryLayer.mlp.dense_4h_to_h.weight']
    mlp_dense_4h_to_h_weight[:,:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.output.projection.weight.npy')
                .transpose()
        ).float()
    mlp_dense_4h_to_h_bias = transformer[f'topQueryLayer.mlp.dense_4h_to_h.bias']
    mlp_dense_4h_to_h_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.output.projection.bias.npy')
        ).float()


    input_layernorm_weight = transformer[f'topQueryLayer.input_layernorm.weight']
    input_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.layernorm1.gamma.npy')
        ).float()
    input_layernorm_bias = transformer[f'topQueryLayer.input_layernorm.bias']
    input_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.layernorm1.beta.npy')
        ).float()

    post_attention_layernorm_weight = transformer[f'topQueryLayer.post_attention_layernorm.weight']
    post_attention_layernorm_weight[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.layernorm2.gamma.npy')
        ).float()
    post_attention_layernorm_bias = transformer[f'topQueryLayer.post_attention_layernorm.bias']
    post_attention_layernorm_bias[:] = \
        torch.tensor(
            np.load(npCkptPath+f'backbone.top_query_layer.layernorm2.beta.npy')
        ).float()


def main():
    # test_split_merge()
    # Args
    args = _parse_args(extra_args_provider=get_change_ckpt_args)
    model_type = args.model_type
    # orig_model_parallel_size = args.model_parallel_size
    args.model_parallel_size = 1
    tokenizer = rebuild_tokenizer(args)

    print('\n merging model parallel partitions ...')
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of tokens ................ {} '.format(
        tokenizer.vocab_size))
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden sise ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(
        args.num_attention_heads))
    print('    maximum position embeddings ..... {}'.format(
        args.max_position_embeddings))

    # Full model.
    print('> building the full model ...')

    mpu.initialize.set_model_parallel_world_size(1)
    mpu.initialize.set_model_parallel_rank(0)

    # initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    merged_model = get_model(model_type)
    iteration = 1000   #any num is ok

    # Save the model.
    args.model_parallel_size = 1
    mpu.initialize.set_model_parallel_rank(0)
    sd = {}
    sd['model'] = merged_model.state_dict_for_save_checkpoint()
    sd['iteration'] = iteration
    merged_path = os.path.join(args.npy_ckpt_path, 'merged')
    checkpoint_name = get_checkpoint_name(merged_path, iteration)
    ensure_directory_exists(checkpoint_name)
    loadModelFromNp(sd, args)
    print('> saving merged model to {}'.format(checkpoint_name))
    torch.save(sd, checkpoint_name)

    recordFile = open(get_checkpoint_tracker_filename(merged_path), 'w')
    recordFile.writelines(str(iteration))
    recordFile.close()


    print('done :-)')

if __name__ == '__main__':

    main()
