import argparse
import yaml
import torch 
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


import numpy as np
import math
import matplotlib.pyplot as plt


from theory import det_eq



def load_yaml(path):
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing {path}: {e}")
    return data

def generate_data(d, nsample, Sigma_diag, beta, tau_te):
 
    # Sigma_diag: (1,d), beta: (d,1)

    assert Sigma_diag.shape == (1,d)
   # B is that batch size
    device = beta.device

    X = torch.randn( nsample, d, dtype = beta.dtype ).to(device) * (torch.sqrt(Sigma_diag.to(device))) #(nsample, d)
    
    y = X @ beta.to(device)  # (nsample, 1)
    eps = math.sqrt(tau_te)*torch.randn(y.size(), dtype = Sigma_diag.dtype, device = device)
    y += eps

    return X, y, eps

def compute_erm(X,  y, reg):

    # X: (nsmaple ,d ), y: ( nsample, 1), reg: 

    nsample, d = X.size(0), X.size(1)

    device = X.device
    y = y.to(device) 

    M = (X.T @ X) + max(reg,1e-12)* torch.eye(d, device = device)#(d,d)

    Xt_y = X.T @ y #(d,1)

    sol = torch.linalg.solve(M, Xt_y) #( d, 1)

    return sol.to(device)

def eval_test(Sigma_diag, beta_hat, beta):

    # Sigma_diag (1,d),  a_hat ( d, 1), beta (d,1)

    device = Sigma_diag.device

    beta = beta.to(device)

    beta_hat = beta_hat.to(device)

    beta = beta.to(device)

    err = torch.sqrt(Sigma_diag.T) * (beta - beta_hat) # (d,1) * (d,1) -> (d,1)

    return torch.sum(err**2)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Load a YAML config file.")
    parser.add_argument("config", type=str, help="Path to YAML config file")

    parser.add_argument("--savepath", type=str, default="./results", help="Path to save results")


    args = parser.parse_args()


    config = load_yaml(args.config)
    print("Loaded config:")
    print(config)

    save_dir = Path(args.savepath)
    save_dir.mkdir(parents=True, exist_ok=True)

    
    d = 10000
    # make n_max \approx 10^5
    scaling_range = np.arange(11.0, 12.6, 0.125)

    num_expr =  50

    test_results = torch.zeros(num_expr, (len(scaling_range )))

    test_deter = torch.zeros( len(scaling_range ))

    ############# Check Bounds ############

    bias_deter = torch.zeros( len(scaling_range ))

    var_deter = torch.zeros( len(scaling_range))

    BQ_deter = torch.zeros( len(scaling_range ))

    B1_deter = torch.zeros( len(scaling_range ))

    BC_deter = torch.zeros( len(scaling_range ))

    ############# Check Bounds ############

    pow_scalingrange = torch.Tensor(np.power(2, scaling_range))

    nsample = pow_scalingrange.long()

    reg = (torch.Tensor(pow_scalingrange)**(-config["gamma_reg_train"]))

    alpha_train = config["alpha_train"]

    r_train = config["r_train"]

    alpha_test = config["alpha_test"]

    r_test = config["r_test"]

    tau = config["tau"]

    Sigma_diag_train = (torch.arange(1, d+1).float() ** (-alpha_train)).unsqueeze(0)
    Sigma_diag_train = Sigma_diag_train.to(device) 

    Sigma_diag_test = (torch.arange(1, d+1).float() ** (-alpha_test)).unsqueeze(0)
    Sigma_diag_test = Sigma_diag_test.to(device) 

    beta_train = (torch.arange(1, d+1, dtype=float) ** (-0.5*(1+2*alpha_train*r_train))).to(Sigma_diag_train.dtype)
    beta_train = beta_train.view(-1,1) 
    beta_train = beta_train.to(device)

    beta_test = (torch.arange(1, d+1, dtype=float) ** (-0.5*(1+2*alpha_test*r_test))).to(Sigma_diag_test.dtype)
    beta_test = beta_test.view(-1,1)
    beta_test= beta_test.to(device)

    print(beta_train[:10])

    print(beta_test[:10])


    for i in range(len(nsample)):

        #bias, var = deterministic_equi_te( nsample_te[i], nfeature_te[i], reg_te[i], Sigma_diag, beta , tau_te  )
        n = nsample[i].item()
        lamb = reg[i].item()
        bias, var, lamb_s, b1, bq, bc_multi = det_eq(Sigma_diag_train, Sigma_diag_test, beta_train, beta_test, n, lamb, tau)

        test_deter[i] = bias + var

        ############# Check Bounds ############

        bias_deter[i] = bias

        var_deter[i] = var

        B1_deter[i] = b1

        BQ_deter[i] = bq

        BC_deter[i] = bc_multi

        ############# Check Bounds ############

        #print(f"bias:{bias}, var:{var}")
    
        for expr in range(num_expr):

            print(f"expr:{expr}, nsample:{nsample[i]}")

            X,y,eps = generate_data(d, n, Sigma_diag_train, beta_train, tau)

            # # #print(X_te.shape, W_te.shape, y_te.shape)
            beta_hat = compute_erm(X,  y, lamb)

            # # #print(a_hat.shape)
            err_te = eval_test(Sigma_diag_test, beta_hat, beta_test)

            # # #print(err.shape)

            test_results[ expr, i] = err_te

            #print(err_te)
         

    res_dict = {
        "config":config,
        "scaling_range":scaling_range,
        "test_res":test_results,
        "test_deter":{ "test_deter": test_deter, "lamb_s": lamb_s } }

    torch.save(res_dict, f"{save_dir}/results.pt")

    ############# Check Bounds ############


    print(B1_deter)
    log2_b1_deter = torch.log2(B1_deter)
    c = (0.2, 0.8, 0.4)
    plt.scatter( scaling_range, log2_b1_deter , marker = 'x', color=c, label="det")
    plt.plot(scaling_range, log2_b1_deter,  color=c )

    print(BQ_deter)
    log2_bq_deter = torch.log2(BQ_deter)
    c = (0.2, 0.4, 0.4)
    plt.scatter( scaling_range, log2_bq_deter , marker = 'x', color=c, label="quad")
    plt.plot(scaling_range, log2_bq_deter,  color=c )

    print(BC_deter)
    log2_bc_deter = torch.log2(BC_deter)
    c = (0.2, 0.4, 0.0)
    plt.scatter( scaling_range, log2_bc_deter , marker = 'x', color=c, label="cros")
    plt.plot(scaling_range, log2_bc_deter,  color=c )

    log2_bias_deter = torch.log2(bias_deter)
    c = (0.2, 0.0, 0.0)
    plt.scatter( scaling_range, log2_bias_deter , marker = 'x', color=c, label="bias")
    plt.plot(scaling_range, log2_bias_deter,  color=c )

    log2_var_deter = torch.log2(var_deter)
    c = (0.2, 0.0, 0.8)
    plt.scatter( scaling_range, log2_var_deter , marker = 'x', color=c, label="var")
    plt.plot(scaling_range, log2_var_deter,  color=c )


    plt.legend()

    plt.savefig(f"{save_dir}/check_bounds.png")

    plt.close()













    








