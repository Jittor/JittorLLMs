import torch

def get_dict(dict1, res):
    if type(dict1) == type({}):
        for key in dict1:
            get_dict(dict1[key], res)
    else:
        try:
            for key in dict1:
                res.append(dict1[key])
        except:
            print(dict1)
            pass

if __name__ == '__main__':
    checkpoint_name = '/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/partitions_splited/merged/iter_0007800/mp_rank_00/model_optim_rng.pt'
    sd1 = torch.load(checkpoint_name, map_location='cpu')

    checkpoint_name = '/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/merged_v3/iter_0001000/mp_rank_00/model_optim_rng.pt'
    sd2 = torch.load(checkpoint_name, map_location='cpu')

    res1 = []
    get_dict(sd1, res1)
    res2 = []
    get_dict(sd2, res2)

    res = []
    for i in range(len(res1)):
        tmp = torch.sub(res1[i].cuda(), res2[i].cuda())
        tmp_max = torch.max(tmp)
        tmp_min = torch.min(tmp)
        res.append(float(tmp_max.cpu()))
        res.append(float(tmp_min.cpu()))
        pass

    a = min(res)
    b = max(res)

    pass