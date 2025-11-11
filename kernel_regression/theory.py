import torch

def self_consistent(lamb, Sigma_diag, n, lamb_in = 1e-2, damp = 0.8, thr = 1e-9, it_max = 5000 ):

    lamb_s = lamb_in

    for it in range(it_max):

        lamb_new = lamb/(n - torch.sum(Sigma_diag/(Sigma_diag + lamb_s)))

        lamb_old = lamb_s

        lamb_s = lamb_old*(1-damp) + lamb_new*damp

        dif = 1-lamb_s/lamb_new

        if torch.abs(dif) <= thr:

            break

        assert lamb_s > 0, 'lambda_s neagtive'

        if it >= it_max:

            print("Max iteration achieved")

    return lamb_s.item()


def det_eq(Sigma_diag_train, Sigma_diag_test, beta_train, beta_test, nsample, lamb, tau):

    # Sigma_diag_train: (1,d), Sigma_diag_test (1,d), beta_train (d,1), beta_test (d,1)

    #print(beta_train - beta_test)

    lamb_s = self_consistent(lamb,Sigma_diag_train,nsample)

    B1 = torch.sum((beta_train - beta_test)*Sigma_diag_test.T *(beta_train - beta_test))

    #print(B1.shape)

    M = 1/(Sigma_diag_train + lamb_s)**2 #(1,d)

    # Quadratic term
    BQ1 = (lamb_s**2) * torch.sum( beta_train * M.T *Sigma_diag_test.T * beta_train )

    #print(BQ1.shape)

    TQ_dnum = torch.sum(Sigma_diag_test * Sigma_diag_train * M)
    TQ_num = torch.sum((Sigma_diag_train**2)*M)
    BQ2 =  (lamb_s**2) * (TQ_dnum/(nsample - TQ_num))*torch.sum(beta_train * Sigma_diag_train.T * M.T * beta_train)

    #print(BQ2.shape)

    # Cross term
    BC  =  2*lamb_s*torch.sum((beta_test - beta_train) * Sigma_diag_test.T * beta_train / (Sigma_diag_train.T + lamb_s))

    BC_multi1 = torch.sum(((beta_test - beta_train) * Sigma_diag_test.T)**2/(Sigma_diag_train.T + lamb_s))

    BC_multi2 = torch.sum( (beta_train)**2 / (Sigma_diag_train.T + lamb_s) )

    BC_multi = 2*lamb_s*torch.sqrt(BC_multi1 * BC_multi2)

    #print(BC.shape)

    Bias = B1 + BQ1 + BQ2 +BC

    BQ = BQ1 + BQ2

    # Variance term

    temp = (Sigma_diag_train) * M
    Var = tau*torch.sum(Sigma_diag_test * temp)/ (nsample - torch.sum(Sigma_diag_train*temp))

    #print( B1.item(), BQ.item(), BC_multi.item())

    return Bias.item(), Var.item(), lamb_s, B1.item(), BQ.item(), BC_multi.item()