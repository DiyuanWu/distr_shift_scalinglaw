import argparse
import yaml
import torch 
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


import matplotlib.pyplot as plt

import numpy as np    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load a results dict.")
    parser.add_argument("path", type=str, help="Path to the results")

    parser.add_argument("--filename", type=str, help="name of the results file")

    args = parser.parse_args()

    path = args.path

    filename = args.filename

    res_dict = torch.load(f"{path}/{filename}")

    scaling_range = res_dict["scaling_range"]

    test_results = res_dict["test_res"]

    test_deter = res_dict["test_deter"]["test_deter"]
    
    print(scaling_range.shape, test_results.shape, test_deter.shape)


    avg_test_te = torch.log2(test_results.detach().cpu().mean(dim=0))

    plt.scatter(scaling_range, avg_test_te, marker = 'o', c='b' ,label="expr")
    
    
    log2_te_deter = torch.log2(test_deter.detach().cpu())

    plt.scatter( scaling_range, log2_te_deter , marker = 'x', c='b', label="detqu")

    plt.plot(scaling_range, log2_te_deter,  c='b' )


    plt.legend()

    plt.savefig(f"{path}/kernel_re.png")

    plt.close()







