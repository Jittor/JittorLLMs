This is a pytorch implementation of the Pangu alpha model. It can be inferred, trained, and finetune on the pytorch framework.

Starting point: Mindspore is a new deep learning framework that many people have not used, so converting mindspore models to pytorch models will allow more people to use our Pangu models and allow users to not only experience our large models, but also finetune our models.

Megatron is a large, powerful transformer algorithm library developed by NVIDIA's deep learning applications research team. This port is based on Megatron, and the main work includes converting model files, adding query layer, and modifying model slicing strategy.

# Environments

Supports python >= 3.6, pytorch >= 1.5, cuda >= 10, and nccl >= 2.6 versions.

The official NVIDIA docker image `docker pull nvcr.io/nvidia/pytorch:20.03-py3` is recommended. You need to install [NLTK](https://www.nltk.org/install.html).

You can also download the paired image directly at

```bash
docker pull yands/pangu-alpha-megatron-lm-nvidia-pytorch:20.03.2
```
Using`/opt/conda/bin/python`。

# Model File Download

| Model File                                                     | Md5                              | Size | Parameter Configuration                                                     |
| ------------------------------------------------------------ | -------------------------------- | ---- | ------------------------------------------------------------ |
| [Pangu-alpha_2.6B_fp16_mgt.zip](https://git.openi.org.cn/attachments/72aec03d-6bdb-4652-ac2a-8099db4b0bed) | 28f6dd2ec5d1df2fd22ec5f4a66f51e7 | 4.6G | num-layers : 31<br />hidden-size : 2560<br />num-attention-heads : 32 |
| [Pangu-alpha_13B_fp16_mgt.zip](https://git.openi.org.cn/attachments/937b3e2d-98fb-4871-9691-b32afb5a4d79?type=0) | e6f7a05cbdf8ba8d69e6786e48344f6f | 22G | num-layers : 39<br />hidden-size : 5120<br />num-attention-heads : 40 |

**Note：`num-layers` is equal to `num-layers - 1` in [Pangu](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)**

Model file directory structure.
```txt
Pangu-alpha_2.6B_fp16_mgt #model directory, --load parameter needs to fill in the path
    -- iter_0001000 # iteration number directory
        --mp_rank_00 # directory for each GPU when the model is parallel
            --model_optim_rng.pt #model file
    --latest_checkpointed_iteration.txt #file of iterations of ckpt
```
# Accuracy
There are some differences in the results of the `mean` operator between the two frameworks, resulting in inconsistent output of the `LayerNorm` layer, so the generated results are not exactly consistent. Not solved yet, looking for a solution :-).

On the iflytek task, the few-shot accuracy of pytorch's 2.6b_fp16 model is 0.78929, which is 2 points down from the paper's 0.81.
# Inference

###** Want a quick experience? Check out the [3-Minute Tutorial on inference](3-minus-inference-en.md)! You can white-knuckle the T4 server!!! **

Currently only the inference script for generating text is available, as follows.

Requires the following configuration parameters.

`-out-seq-length`: the maximum number of tokens to generate

`--top_k`: the larger the value of k, the higher the diversity of generated samples. Different k can be tried.

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
Examples：

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
Currently only finetune is provided without changing the model model structure and data format, i.e. continue pre-training.
##### 1. Preparing training data

Refer to [data](#data) section

##### 2. Model cutting

The model downloaded above is a single machine inference model, so you need to cut the model first when finetune is performed, and cut it into model parallel models.

Parameters.

`-model-parallel-size`: the number of slices of the original model, here is 1

`--num-mp-model`: the number of models after slicing

`--mp-model-save`: the path to save the model after slicing

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
##### 3. Training

Run the script:

```examples/finetune_pangu_distributed.sh```

##### 4. Model merging

The finished model of finetune is fragmented, so if you want to do single card inference, you need to merge the model first.

Merge script.

`--mp-model-parallel-size`: the number of model slices

`--load`: model save directory

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



# Training

Reference Script

```bash
examples/pretrain_pangu_distributed_2.6B.sh
```



# Data

##### Generate training data

Reference script: `/tools/preprocess_data_pangu.py`

Store multiple `xxx.txt` files in the train_dataset directory, if there are more training data, it is better to have a uniform file size for each `txt` and separate multiple `txt`s, the size can be 10M a file. If there is traditional text that needs to be converted to simplified, you can use `zhconv`.

The format of each `txt` text is (need blank lines to split different samples)
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

The files /path/to/dataset/xxx.idx and /path/to/dataset/xxx.bin will be generated.

Finetune and pre-training require the parameter: `-data-path=/path/to/dataset/xxx`





