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
import copy

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

def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def split_into_partitions(tensor, num_partitions, partition_dim):

    per_partition_size = mpu.utils.divide(tensor.size(partition_dim),
                                          num_partitions)

    partitions_list = torch.split(tensor,
                                  per_partition_size,
                                  dim=partition_dim)

    return partitions_list


def get_parallel_checkpoint_name(path):

    tracker_filename = get_checkpoint_tracker_filename(path)
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)
    assert iteration > 0
    checkpoint_name = get_checkpoint_name(path, iteration)

    return checkpoint_name, iteration


def splitTensorlIntoPartition(tensor, tensorNewList, mpPartitions, tensorName, dim):
    tensor = tensor[tensorName]
    partitionsList = split_into_partitions(tensor, mpPartitions, dim)
    for i, partition in enumerate(partitionsList):
        tensorNewList[i][tensorName] = partition

def loadPartionEmbending(model, modelNewList, mpPartitions):

    embedding = model['model']['language_model']['embedding']['word_embeddings']
    embeddingNewList = [modelNew['model']['language_model']['embedding']['word_embeddings']
                        for modelNew in modelNewList]
    tensorName = f'weight'

    if mpPartitions == 2:
        embeddingSizePerPartition = 20096

    a = embedding[tensorName]
    newSize = embeddingSizePerPartition*mpPartitions
    b = torch.rand(newSize-a.shape[0], a.shape[1])

    embedding[tensorName] = torch.cat((a,b),0)
    splitTensorlIntoPartition(embedding, embeddingNewList, mpPartitions, tensorName, 0)



def loadPartionAttentionLayer(model, modelNewList, mpPartitions,layerID):

    transformer = model['model']['language_model']['transformer']
    transformerNewList = [modelNew['model']['language_model']['transformer']
                          for modelNew in modelNewList]

    tensorName = f'layers.{layerID}.attention.query.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'layers.{layerID}.attention.query.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'layers.{layerID}.attention.key.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'layers.{layerID}.attention.key.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'layers.{layerID}.attention.value.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'layers.{layerID}.attention.value.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'layers.{layerID}.attention.dense.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 1)
    # tensorName = f'layers.{layerID}.attention.dense.bias'
    # splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'layers.{layerID}.mlp.dense_h_to_4h.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'layers.{layerID}.mlp.dense_h_to_4h.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'layers.{layerID}.mlp.dense_4h_to_h.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 1)
    # tensorName = f'layers.{layerID}.mlp.dense_4h_to_h.bias'
    # splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)


def loadPartionTopQueryLayer(model, modelNewList, mpPartitions):

    transformer = model['model']['language_model']['transformer']
    transformerNewList = [modelNew['model']['language_model']['transformer']
                          for modelNew in modelNewList]

    tensorName = f'topQueryLayer.attention.query.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'topQueryLayer.attention.query.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'topQueryLayer.attention.key.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'topQueryLayer.attention.key.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'topQueryLayer.attention.value.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'topQueryLayer.attention.value.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'topQueryLayer.attention.dense.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 1)
    # tensorName = f'layers.{layerID}.attention.dense.bias'
    # splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'topQueryLayer.mlp.dense_h_to_4h.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)
    tensorName = f'topQueryLayer.mlp.dense_h_to_4h.bias'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)

    tensorName = f'topQueryLayer.mlp.dense_4h_to_h.weight'
    splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 1)
    # tensorName = f'layers.{layerID}.mlp.dense_4h_to_h.bias'
    # splitTensorlIntoPartition(transformer, transformerNewList, mpPartitions, tensorName, 0)


def main():
    mpPartitions = 2
    numLayers = 31

    partitionId = 0
    ckptName = '/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/merged_v2/iter_0007800/mp_rank_00/model_optim_rng.pt'
    model = torch.load(ckptName)
    transformerNewList = []
    modelNewList = [copy.deepcopy(model) for i  in range(mpPartitions)]

    loadPartionEmbending(model, modelNewList, mpPartitions)

    for layerID in range(numLayers):
        loadPartionAttentionLayer(model, modelNewList, mpPartitions, layerID)

    loadPartionTopQueryLayer(model, modelNewList, mpPartitions)

    save = '/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/merged_split'
    iteration = 'iter_0007800'
    modelName = 'model_optim_rng.pt'
    for rank, model_ in enumerate(modelNewList):
        rankName = 'mp_rank_{:02d}'.format(rank)
        savePath='/'.join([save,iteration,rankName,modelName])
        ensure_directory_exists(savePath)
        tensorName = 'layers.0.attention.query.weight'
        print(model_['model']['language_model']['transformer'][tensorName].shape)
        torch.save(model_, savePath)

    pass



if __name__ == '__main__':

    main()
