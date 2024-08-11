import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


def fourier_basis(x, p):
    res = []
    for i in range(0, p):
        k = (i + 1) // 2
        if i == 0:
            res.append(1 / np.sqrt(2 * np.pi))
        elif (i + 1) % 2 == 0:
            res.append(1 / np.sqrt(np.pi) * np.cos(k * x))
        else:
            res.append(1 / np.sqrt(np.pi) * np.sin(k * x))
    return np.array(res)


def get_fourier_basis(x, p):
    res = np.zeros((x.shape[0], p))
    for i in range(x.shape[0]):
        res[i, :] = fourier_basis(x[i], p)
    return res


def get_data(d_x=10, d_x_overfit=20, n_data=50, x_start=-1.0):
    """
    data generating process:v(y|x)=p(y|x,theta)=N(fourier(x)^T@theta,variance), where theta=1
    """
    variance = 1.0
    d_y = 1

    X_orig = np.random.uniform(x_start, x_start + 2, n_data)
    X = np.asarray(get_fourier_basis(X_orig, d_x))
    X_overfit = np.asarray(get_fourier_basis(X_orig, d_x_overfit))
    Y = np.random.normal(
        np.sum(X, 1).reshape(n_data, 1), np.sqrt(variance), (n_data, 1)
    )

    assert X.shape == (n_data, d_x)
    assert Y.shape == (n_data, d_y)
    assert X_orig.shape == (n_data,)

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(X_overfit).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(X_orig).float(),
    )


def get_X_show(d_x=10, d_x_overfit=20, n_data=10000):
    """handy for making model fit plot"""
    X_orig = np.linspace(-1.0, 1.0, n_data)
    X = np.asarray(get_fourier_basis(X_orig, d_x))
    X_overfit = np.asarray(get_fourier_basis(X_orig, d_x_overfit))

    assert X.shape == (n_data, d_x)
    assert X_orig.shape == (n_data,)

    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(X_overfit).float(),
        torch.from_numpy(X_orig).float(),
    )


def get_metrics(
    X_train,
    Y_train,
    X_test,
    Y_test,
    var_prior,
    var_likelihood,
    lambs,
    n_post_samples=10000,
):
    results = {}

    results["p_post"] = []

    results["nll_bayes_test"] = []
    results["nll_bayes_train"] = []
    results["nll_gibbs_test"] = []
    results["nll_gibbs_train"] = []

    results["mse_bayes_test"] = []
    results["mse_bayes_train"] = []
    results["mse_gibbs_test"] = []
    results["mse_gibbs_train"] = []

    results["grad_expected_bayes"] = []
    results["grad_empirical_gibbs"] = []

    for lamb in lambs:
        print(lamb)

        d_x = X_train.shape[1]

        # prior
        var_prior = torch.tensor(var_prior)
        # likelihood
        var_likelihood = torch.tensor(var_likelihood)

        # compute posterior distribution, see bishop eq 3.53 and 3.54
        # X_train is the design matrix, n_train by d_x matrix
        alpha = var_prior ** (-1)  # prior recision
        beta_orig = var_likelihood ** (
            -1
        )  # gaussian likelihood precision before absorbing temperature
        beta = (
            beta_orig * lamb
        )  # gaussian likelihood precision after absorbing temperature
        S_N_inv = (
            alpha * torch.eye(d_x) + beta * X_train.T @ X_train
        )  # posterior precision
        S_N = torch.inverse(S_N_inv)  # posterior variance
        m_N = (beta * S_N @ X_train.T @ Y_train).reshape(-1)  # posterior mean
        p_post = MultivariateNormal(m_N, precision_matrix=S_N_inv)
        results["p_post"].append(p_post)

        # X:n,d_x, Y:n,1
        # samples_post:n_post_samples,d_x
        # log_p:n,n_post_samples
        samples_post = p_post.sample((n_post_samples,))
        log_p_test = Normal(
            X_test @ samples_post.T, torch.sqrt(var_likelihood)
        ).log_prob(Y_test)
        log_p_train = Normal(
            X_train @ samples_post.T, torch.sqrt(var_likelihood)
        ).log_prob(Y_train)

        # compute nll
        nll_bayes_test = (
            (torch.log(torch.tensor(n_post_samples)) - torch.logsumexp(log_p_test, 1))
            .mean()
            .item()
        )
        results["nll_bayes_test"].append(nll_bayes_test)

        nll_bayes_train = (
            (torch.log(torch.tensor(n_post_samples)) - torch.logsumexp(log_p_train, 1))
            .mean()
            .item()
        )
        results["nll_bayes_train"].append(nll_bayes_train)

        nll_gibbs_test = -log_p_test.mean().item()
        results["nll_gibbs_test"].append(nll_gibbs_test)

        nll_gibbs_train = -log_p_train.mean().item()
        results["nll_gibbs_train"].append(nll_gibbs_train)

        # compute mse
        mse_bayes_test = (
            (((X_test @ samples_post.mean(0)) - Y_test.reshape(-1)) ** 2).mean().item()
        )
        mse_bayes_train = (
            (((X_train @ samples_post.mean(0)) - Y_train.reshape(-1)) ** 2)
            .mean()
            .item()
        )
        mse_gibbs_test = (((X_test @ samples_post.T) - Y_test) ** 2).mean().item()
        mse_gibbs_train = (((X_train @ samples_post.T) - Y_train) ** 2).mean().item()

        results["mse_bayes_test"].append(mse_bayes_test)
        results["mse_bayes_train"].append(mse_bayes_train)
        results["mse_gibbs_test"].append(mse_gibbs_test)
        results["mse_gibbs_train"].append(mse_gibbs_train)

        # compute grad
        log_p_train.sum(0)
        torch.exp(log_p_test)
        p_test = torch.exp(log_p_test)
        grad_expected_bayes = (
            -1
            * (
                (
                    (p_test * log_p_train.sum(0)).mean(1)
                    - log_p_train.sum(0).mean() * p_test.mean(1)
                )
                / p_test.mean(1)
            )
            .mean()
            .item()
        )
        results["grad_expected_bayes"].append(grad_expected_bayes)
        grad_empirical_gibbs = -torch.var(log_p_train.sum(0)).item()
        results["grad_empirical_gibbs"].append(grad_empirical_gibbs)

    return results


def get_post(X_train, Y_train, var_prior, var_likelihood, lamb):
    d_x = X_train.shape[1]

    var_prior = torch.tensor(var_prior)
    var_likelihood = torch.tensor(var_likelihood)

    # compute posterior distribution, see bishop eq 3.53 and 3.54
    # X_train is the design matrix, n_train by d_x matrix
    alpha = var_prior ** (-1)  # prior recision
    beta_orig = var_likelihood ** (
        -1
    )  # gaussian likelihood precision before absorbing temperature
    beta = beta_orig * lamb  # gaussian likelihood precision after absorbing temperature
    S_N_inv = alpha * torch.eye(d_x) + beta * X_train.T @ X_train  # posterior precision
    S_N = torch.inverse(S_N_inv)  # posterior variance
    m_N = (beta * S_N @ X_train.T @ Y_train).reshape(-1)  # posterior mean

    return m_N, S_N, MultivariateNormal(m_N, precision_matrix=S_N_inv)


def get_grad_emp_gibbs(p_post, n_post_samples, var_likelihood, X_train, Y_train):
    post_samples = p_post.sample(sample_shape=(n_post_samples,))
    likelihood = Normal(
        X_train @ post_samples.T, torch.sqrt(torch.tensor(var_likelihood))
    )
    return -torch.var(
        likelihood.log_prob(Y_train.reshape((Y_train.shape[0], 1))).sum(0)
    )
