import torch

if __name__ == '__main__':

    full_model = torch.load(f"/userhome/tmp/mp_test/"
    f"model parallel {1}"
    f"_rank{0}"
    f"_lyaer{0}"
    f"_hidden_states.pth")

    rank0_model = torch.load(f"/userhome/tmp/mp_test/"
                            f"model parallel {2}"
                            f"_rank{0}"
                            f"_lyaer{0}"
                            f"_hidden_states.pth")

    rank1_model = torch.load(f"/userhome/tmp/mp_test/"
                             f"model parallel {2}"
                             f"_rank{1}"
                             f"_lyaer{0}"
                             f"_hidden_states.pth")

    print(full_model[0])
    print(rank0_model[0])
    print(rank1_model[0])

    pass