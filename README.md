# The Cold Posterior Eect Indicates Underfitting, and Cold Posteriors Represent a Fully Bayesian Method to Mitigate It
This repo provides the codes for reproducing the experiments in the paper [The Cold Posterior Eect Indicates Underfitting, and Cold Posteriors Represent a Fully Bayesian Method to Mitigate It](https://openreview.net/forum?id=GZORXGxHHT&referrer=%5Bthe%20profile%20of%20Yijie%20Zhang%5D(%2Fprofile%3Fid%3D~Yijie_Zhang1)). This implementation is built on this [repo](https://github.com/activatedgeek/understanding-bayesian-classification). 

## Setup
All requirements are listed in [environment.yml](https://github.com/pyijiezhang/cpe-underfit/blob/main/environment.yml). Create a conda environment using:

```
conda env create -n <env_name>
```

Next, ensure Python modules under the src folder are importable as,

```
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```
## Usage
### Neural networks with approximate inference
The main script to run all SGMCMC experiments is [experiments/train_lik.py](https://github.com/pyijiezhang/cpe-underfit/blob/main/experiments/train_lik.py).

As an example, to run cyclical SGHMC with our proposed noisy Dirichlet likelihood on CIFAR-10 with label noise, run:
```shell
python experiments/train_lik.py --dataset=cifar10 \
                                --dirty_lik=resnet18std \
                                --likelihood=softmax \
                                --augment=False \
                                --perm=False \
                                --likelihood_temp=0.5 \
                                --temperature=1.0 \
                                --logits_temp=1.0 \
                                --prior-scale=0.0005 \
                                --sgld-epochs=1000 \
                                --sgld-lr=2e-7
```

Each argument to the `main` method can be used as a command line argument due to [Fire](https://google.github.io/python-fire/guide/).
[Weights & Biases](https://docs.wandb.ai) is used for all logging.

### Linear models with exact inference
The results of linear models can be reproduced in [blm/blm_regression_eact.ipynb](https://github.com/pyijiezhang/cpe-underfit/blob/main/blm/blm_regression_eact.ipynb).
## Bibtex
It will be updated soon.
