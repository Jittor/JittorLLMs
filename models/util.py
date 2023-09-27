import os


def download_fromhub(path, tdir=""):
    import jittor as jt
    from jittor import compiler

    tdir = tdir + '/' if tdir != "" else ""
    if path.startswith("jittorhub://"):
        path = path.replace(
            "jittorhub://", f"https://cg.cs.tsinghua.edu.cn/jittor/{tdir}assets/build/checkpoints/")
        base = path.split("/")[-1]
        fname = os.path.join(compiler.ck_path, tdir, base)
        from jittor_utils.misc import download_url_to_local
        if not os.path.exists(fname):
            download_url_to_local(path, base, os.path.join(
                compiler.ck_path, tdir), None)
        return fname


def get_chatrwkv():
    return download_fromhub("jittorhub://RWKV-4-Pile-3B-EngChn-test4-20230115-fp32.pth", tdir="ChatRWKV")


def get_llama():
    llama_file_list = [
        'llama_consolidated.00.pth',
        'params.json',
        'tokenizer.model'
    ]
    new_paths = []
    for f in llama_file_list:
        new_paths.append(download_fromhub(f"jittorhub://{f}", tdir="llama"))
    return new_paths

def get_llama2():
    llama_file_list = [
        'llama_consolidated.00.pth',
        'params.json',
        'tokenizer.model'
    ]
    new_paths = []
    for f in llama_file_list:
        new_paths.append(download_fromhub(f"jittorhub://{f}", tdir="llama2"))
    return new_paths

def get_atom7b():
    pass

def get_chatglm():
    chatglm_file_list = [
        'pytorch_model-00005-of-00008.bin',
        'pytorch_model-00006-of-00008.bin',
        'pytorch_model-00007-of-00008.bin',
        'pytorch_model-00008-of-00008.bin',
        'pytorch_model-00001-of-00008.bin',
        'pytorch_model-00002-of-00008.bin',
        'pytorch_model-00003-of-00008.bin',
        'pytorch_model-00004-of-00008.bin'
    ]
    new_path = []
    model_dir = os.path.dirname(os.path.realpath(__file__))
    for f in chatglm_file_list:
        new_path.append(download_fromhub(f"jittorhub://{f}", tdir="chat-glm"))
        ln_dir = os.path.join(model_dir, "chatglm")
        if not os.path.exists(os.path.join(ln_dir, f)):
            os.symlink(new_path[-1], os.path.join(ln_dir, f))
    return new_path


def get_pangualpha():
    path = download_fromhub(f"jittorhub://model_optim_rng.pth", tdir="pangu")
    pdir = os.path.dirname(path)
    if not os.path.exists(os.path.join(pdir, "Pangu-alpha_2.6B_fp16_mgt")):
        model_dir = os.path.join(
            pdir, "Pangu-alpha_2.6B_fp16_mgt", "iter_0001000", "mp_rank_00")
        os.makedirs(model_dir)
        f = open(os.path.join(pdir, "Pangu-alpha_2.6B_fp16_mgt",
                 "latest_checkpointed_iteration.txt"), "w")
        f.write(str(1000)+"\n")
        f.close()
        os.symlink(path, os.path.join(model_dir, "model_optim_rng.pth"))
