这是盘古α模型的 pytorch 实现版本。可以在 pytorch 框架上进行推理、训练、finetune。

出发点：Mindspore 是新的深度学习框架，很多人没用过，所以把 mindspore 模型转成 pytorch 模型可以让更多人使用我们的盘古模型，让使用者不但可以体验我们的大模型，还可以对我们的模型进行 finetune 。

Megatron 是英伟达深度学习应用研究团队开发的一款大型、强大的 transformer 算法库。这次的移植是在 Megatron 的基础上修改得到，主要工作内容包括了模型文件的转换、增加 query layer、修改模型切分策略。

# 环境

支持 python >= 3.6, pytorch >= 1.5, cuda >= 10, and nccl >= 2.6 版本.

推荐使用英伟达的官方 docker 镜像`docker pull nvcr.io/nvidia/pytorch:20.03-py3`。需要安装 [NLTK](https://www.nltk.org/install.html)。

也可直接下载配好的镜像：

```bash
docker pull yands/pangu-alpha-megatron-lm-nvidia-pytorch:20.03.2
```
使用`/opt/conda/bin/python`。

# 模型文件下载

| 模型文件                                                     | Md5                              | 大小 | 参数配置                                                     |
| ------------------------------------------------------------ | -------------------------------- | ---- | ------------------------------------------------------------ |
| [Pangu-alpha_2.6B_fp16_mgt.zip](https://git.openi.org.cn/attachments/72aec03d-6bdb-4652-ac2a-8099db4b0bed) | 28f6dd2ec5d1df2fd22ec5f4a66f51e7 | 4.6G | num-layers : 31<br />hidden-size : 2560<br />num-attention-heads : 32 |
| [Pangu-alpha_13B_fp16_mgt.zip](https://git.openi.org.cn/attachments/937b3e2d-98fb-4871-9691-b32afb5a4d79?type=0) | e6f7a05cbdf8ba8d69e6786e48344f6f | 22G | num-layers : 39<br />hidden-size : 5120<br />num-attention-heads : 40 |

**注：`num-layers` 等于 [Pangu](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 项目中的 `num-layers - 1`**

模型文件目录结构：
```txt
Pangu-alpha_2.6B_fp16_mgt                       #模型目录，--load 参数需要填写的路径
    -- iter_0001000                             #迭代次数目录
        --mp_rank_00                            #模型并行时各个 GPU 的目录
            --model_optim_rng.pt                #模型文件
    --latest_checkpointed_iteration.txt         #记录 ckpt 的迭代次数文件
```
# 精度
两个框架的`mean`算子的结果有一定的差异，导致 `LayerNorm` 层的输出不一致，所以生成结果不完全一致。暂时还没解决，正在寻找解决方案 :-) 。

在 iflytek 任务上，pytorch 的 2.6b_fp16 模型的 few-shot 精度为0.78929，相对于论文的的0.81 下降了 2 个点。
# 推理

###**想快速体验？？请查看[3分钟实现推理教程](3-minus-inference.md)！！可以白嫖 T4 服务器哦！！！**

目前只有生成文本的推理脚本，如下：

需要配置参数：

`--out-seq-length`：生成的最大 token 数

`--top_k`：k 值越大生成样本多样性越高。可以尝试不同的 k。

```bash
python tools/generate_samples_Pangu.py \
--model-parallel-size 1 \
--num-layers 31 \
--hidden-size 2560 \
--load /dataset/Pangu-alpha_2.6B_mgt/ \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--out-seq-length 50 \
--temperature 1.0 \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--num-samples 0 \
--top_k 2 \
--finetune
```
例子：

k=1
```txt
Input is: 书生：羌笛何须怨杨柳，春风不度玉门关。
飞云：（这诗怎么这么耳熟？且过去跟他聊聊如何。）
书生：小兄弟，要不要一起喝一杯？
飞云：你请我呀？你若是请我，我便和你喝一杯；你若不请我，我便一个人去喝。
书生：小兄弟，看你年纪轻轻，不至于这么势利吧？
飞云：
Output is: 我是个读书人,不势利,我只想和你喝酒。
书生:小兄弟,你是不是有什么心事?

Input is: 张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，
Output is: 张无忌道:“你这是做甚么?”赵敏道:“我要你做我的丈夫。”张无忌道:“我不配。”赵敏道:“你不配,我也不配

Input is: 四川的省会是?
Output is: 
成都,简称蓉,别称“锦城”,是四川省省会,也是中国西部地区重要的中心城市,中国西部地区重要的经济中心、科技中心、金融中心和交通枢纽,中国西部地区重要的
```
k=2
```txt
Input is: 书生：羌笛何须怨杨柳，春风不度玉门关。
飞云：（这诗怎么这么耳熟？且过去跟他聊聊如何。）
书生：小兄弟，要不要一起喝一杯？
飞云：你请我呀？你若是请我，我便和你喝一杯；你若不请我，我便一个人去喝。
书生：小兄弟，看你年纪轻轻，不至于这么势利吧？
飞云：
Output is: 你这是在质疑本公子的人格?你若是不请我喝酒我就不去喝。你若不请我喝,那我就一个人走路。你要是不请我喝,那你便
```
k = 3
```txt
Input is: 中国和美国和日本和法国和加拿大和澳大利亚的首都分别是哪里？
Output is: 分别在哪个省的哪个市
中国的北京,美国纽约,加拿大多伦多,日本的大阪,法国的里昂,澳大利亚的墨尔本,新西兰的基督城,澳大利亚首都堪
```


# Finetune
目前只提供了不改变模型模型结构和数据格式的 finetune，也就是继续预训练。
##### 1、准备训练数据

参考[数据](#数据)部分

##### 2、模型切割

上面下载的模型是单机推理模型，所以在进行 finetune 的时候需要先对模型进行切割，切割成模型并行的模型。

参数：

`model-parallel-size`：原始模型的分片个数，这里是 1

`--num-mp-model`： 切分后的模型个数

`--mp-model-save`：切分后，模型的保存路径

```bash
python tools/split_full_model_into_mp_model.py \
--model-parallel-size 1 \
--num-mp-model 2 \
--num-layers 31 \
--hidden-size 2560 \
--load /**ful model path**/ \
--mp-model-save /**mp model save path**/ \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--model-type Pangu \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--finetune
```

##### 3、训练

运行脚本:

```examples/finetune_pangu_distributed.sh```

##### 4、模型合并

finetune 完后的模型是分片的，如果要进行单卡推理，则先需要对模型进行合并。

合并脚本：

`--mp-model-parallel-size`：模型分片数

`--load`：模型保存目录

```bash
python tool/merge_mp_partitions.py \
--model-parallel-size 1 \
--mp-model-parallel-size 2 \
--num-layers 31 \
--hidden-size 2560 \
--load /full model ckpt dir/  \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--model-type Pangu \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--reset-attention-mask \
--finetune \
```



# 训练

参考脚本

```bash
examples/pretrain_pangu_distributed_2.6B.sh
```



# 数据

##### 生成训练数据

参考脚本：`/tools/preprocess_data_pangu.py`

在 train_dataset 目录下存放多个 `xxx.txt` 文件，如果训练数据较多，最好每个 `txt` 文件大小统一，且分开多个 `txt` 存放，大小可以 10M 一个文件。如果有繁体文字，需要转成简体，可以使用`zhconv`。

每个 `txt` 文本格式为（需要空行分割不同样本）：
```txt
sample 1 ***
***
***

sample 2 ***
***
***

sample 2 ***
***
***
```
```bash
python /tools/preprocess_data_pangu.py \
--input /train_dataset/*.txt \
--output-prefix /megatron/dataset/ \
--vocab-file /megatron/tokenizer/bpe_4w_pcl/vocab \
--dataset-impl mmap \
--append-eod
```

将会生成/path/to/dataset/xxx.idx 和 /path/to/dataset/xxx.bin 文件。

Finetune 和预训练时需要填写参数：`--data-path=/path/to/dataset/xxx`





