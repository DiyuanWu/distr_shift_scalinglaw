

## Configuration

The script reads a YAML file (positional `config`) with the following keys:

```yaml
# Example: config/example.yaml
gamma_reg_train: 0.5   # exponent for training ridge penalty: lambda = (2^k)^(-gamma)
alpha_train: 1.0       # train covariance power exponent (Sigma_ii ∝ i^(-alpha_train))
r_train: 0.0           # train signal smoothness exponent
alpha_test: 1.0        # test covariance power exponent
r_test: 0.0            # test signal smoothness exponent
tau: 0.1               # noise variance
```

**Model family used in the script**

* Covariance (train/test): $\Sigma_{ii} = i^{-\alpha_{\cdot}}$
* Signal (train/test): $\beta_i \propto i^{-\tfrac{1}{2}(1+2\alpha r_{\cdot})}$
* Regularization : $\lambda = (2^k)^{-\gamma}$

---


## Usage

```bash
python your_script.py config/example.yaml --savepath ./results
```

**Arguments**

* `config` (positional): path to YAML config
* `--savepath` (optional, default `./results`): output directory

**Outputs**

* `results.pt` (PyTorch pickle) containing:

  ```python
  {
    "config": <dict>,                
    "scaling_range": <np.ndarray>,  
    "test_res": <tensor [n_expr, n_scaling]>,    
    "test_deter": {
        "test_deter": <tensor [n_scaling]>,  
        "lamb_s": <...>              
    }
  }
  ```

---


Specify your license (e.g., MIT/Apache-2.0/GPL) here. If you’re unsure, MIT is a permissive default for open-source code.
