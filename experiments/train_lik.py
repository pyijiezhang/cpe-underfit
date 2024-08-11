import logging
from pathlib import Path
from tqdm.auto import tqdm
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_aug.optim import SGLD
from data_aug.optim.lr_scheduler import CosineLR
from data_aug.utils import set_seeds
from data_aug.models import (
    ResNet18,
    ResNet18FRN,
    ResNet18Fixup,
    LeNetLarge,
    LeNetSmall,
    MLP,
)
from data_aug.datasets import (
    get_cifar10,
    get_cifar100,
    get_tiny_imagenet,
    get_mnist,
    get_fmnist,
    prepare_transforms,
)
from data_aug.nn import (
    GaussianPriorAugmentedCELoss,
    KLAugmentedNoisyDirichletLoss,
    NoisyDirichletLoss,
)


def calibration_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10,
    top_class_only: bool = True,
    equal_size_bins: bool = False,
    min_p: float = 0.0,
):
    if probabilities.ndim == targets.ndim + 1:
        # multi-class
        if top_class_only:
            # targets are converted to per-datapoint accuracies, i.e. checking whether or not the predicted
            # class was observed
            predictions = np.cast[targets.dtype](probabilities.argmax(-1))
            targets = targets == predictions
            probabilities = probabilities.max(-1)
        else:
            # convert the targets to one-hot encodings and flatten both those targets and the probabilities,
            # treating them as independent predictions for binary classification
            num_classes = probabilities.shape[-1]
            one_hot_targets = np.cast[targets.dtype](
                targets[..., np.newaxis] == np.arange(num_classes)
            )
            targets = one_hot_targets.reshape(*targets.shape[:-1], -1)
            probabilities = probabilities.reshape(*probabilities.shape[:-2], -1)

    elif probabilities.ndim != targets.ndim:
        raise ValueError(
            "Shapes of probabilities and targets do not match. "
            "Must be either equal (binary classification) or probabilities "
            "must have exactly one dimension more (multi-class)."
        )
    else:
        # binary predictions, no pre-processing to do
        pass

    if equal_size_bins:
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(probabilities, quantiles)
        # explicitly set upper and lower edge to be 0/1
        bin_edges[0] = 0
        bin_edges[-1] = 1
    else:
        bin_edges = np.linspace(0, 1, num_bins + 1)

    # bin membership has to be checked with strict inequality to either the lower or upper
    # edge to avoid predictions exactly on a boundary to be included in multiple bins.
    # Therefore the exclusive boundary has to be slightly below or above the actual value
    # to avoid 0 or 1 predictions to not be assigned to any bin
    bin_edges[0] -= 1e-6
    lower = bin_edges[:-1]
    upper = bin_edges[1:]
    probabilities = probabilities.reshape(-1, 1)
    targets = targets.reshape(-1, 1)

    # set up masks for checking which bin probabilities fall into and whether they are above the minimum
    # threshold. I'm doing this by multiplication with those booleans rather than indexing in order to
    # allow for the code to be extensible for broadcasting
    bin_membership = (probabilities > lower) & (probabilities <= upper)
    exceeds_threshold = probabilities >= min_p

    bin_sizes = (bin_membership * exceeds_threshold).sum(-2)
    non_empty = bin_sizes > 0

    bin_probability = np.full(num_bins, np.nan)
    np.divide(
        (probabilities * bin_membership * exceeds_threshold).sum(-2),
        bin_sizes,
        out=bin_probability,
        where=non_empty,
    )

    bin_frequency = np.full(num_bins, np.nan)
    np.divide(
        (targets * bin_membership * exceeds_threshold).sum(-2),
        bin_sizes,
        out=bin_frequency,
        where=non_empty,
    )

    bin_weights = np.zeros(num_bins)
    np.divide(bin_sizes, bin_sizes.sum(), out=bin_weights, where=non_empty)

    return bin_probability, bin_frequency, bin_weights


def expected_calibration_error(
    mean_probability_predicted: np.ndarray,
    observed_frequency: np.ndarray,
    bin_weights: np.ndarray,
):
    """Calculates the ECE, i.e. the average absolute difference between predicted probabilities and
    true observed frequencies for a classifier and its targets. Inputs are expected to be formatted
    as the return values from the calibration_curve method. NaNs in mean_probability_predicted and
    observed_frequency are ignored if the corresponding entry in bin_weights is 0."""
    idx = bin_weights > 0
    return np.sum(
        np.abs(mean_probability_predicted[idx] - observed_frequency[idx])
        * bin_weights[idx]
    )


def average_l2_norm(model, num_params):
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.norm(2)  # Calculate L2 norm for the parameter
            total_norm += param_norm.item() ** 2  # Add the squared norm to the total

    average_norm = (
        total_norm / num_params
    ) ** 0.5  # Calculate the square root to get the L2 norm
    return average_norm


@torch.no_grad()
def test(data_loader, net, criterion, device=None):
    net.eval()

    total_loss = 0.0
    N = 0
    Nc = 0

    for X, Y in tqdm(data_loader, leave=False):
        X, Y = X.to(device), Y.to(device)

        f_hat = net(X)
        Y_pred = f_hat.argmax(dim=-1)
        loss = criterion(f_hat, Y, N=X.size(0))

        N += Y.size(0)
        Nc += (Y_pred == Y).sum().item()
        total_loss += loss

    acc = Nc / N

    return {
        "total_loss": total_loss.item(),
        "acc": acc,
    }


@torch.no_grad()
def test_bma(net, data_loader, samples_dir, nll_criterion=None, device=None):
    net.eval()

    ens_logits = []
    ens_nll = []

    for sample_path in tqdm(Path(samples_dir).rglob("*.pt"), leave=False):
        net.load_state_dict(torch.load(sample_path))

        all_logits = []
        all_Y = []
        all_nll = torch.tensor(0.0).to(device)
        for X, Y in tqdm(data_loader, leave=False):
            X, Y = X.to(device), Y.to(device)
            _logits = net(X)
            all_logits.append(_logits)
            all_Y.append(Y)
            if nll_criterion is not None:
                all_nll += nll_criterion(_logits, Y)
        all_logits = torch.cat(all_logits)
        all_Y = torch.cat(all_Y)

        ens_logits.append(all_logits)
        ens_nll.append(all_nll)

    ens_logits = torch.stack(ens_logits)
    ens_nll = torch.stack(ens_nll)

    ce_nll = (
        -torch.distributions.Categorical(logits=ens_logits)
        .log_prob(all_Y)
        .sum(dim=-1)
        .mean(dim=-1)
    )

    nll = ens_nll.mean(dim=-1)

    Y_pred = ens_logits.softmax(dim=-1).mean(dim=0).argmax(dim=-1)
    acc = (Y_pred == all_Y).sum().item() / Y_pred.size(0)

    return {"acc": acc, "nll": nll, "ce_nll": ce_nll}


@torch.no_grad()
def get_metrics_training(net, logits_temp, data_loader, device=None):
    net.eval()

    all_logits = []
    all_Y = []
    for X, Y in tqdm(data_loader, leave=False):
        X, Y = X.to(device), Y.to(device)
        _logits = net(X)
        all_logits.append(_logits)
        all_Y.append(Y)
    all_logits = torch.cat(all_logits)
    all_logits.div_(logits_temp)
    all_Y = torch.cat(all_Y)

    log_p = torch.distributions.Categorical(logits=all_logits).log_prob(all_Y)
    Y_pred = all_logits.softmax(dim=-1).argmax(dim=-1)
    acc = (Y_pred == all_Y).sum().item() / Y_pred.size(0)
    return log_p, acc


@torch.no_grad()
def get_metrics_bma(net, logits_temp, data_loader, samples_dir, device=None):
    net.eval()

    ens_logits = []
    for sample_path in tqdm(Path(samples_dir).rglob("*.pt"), leave=False):
        net.load_state_dict(torch.load(sample_path))

        all_logits = []
        all_Y = []
        for X, Y in tqdm(data_loader, leave=False):
            X, Y = X.to(device), Y.to(device)
            _logits = net(X)
            all_logits.append(_logits)
            all_Y.append(Y)
        all_logits = torch.cat(all_logits)
        all_logits.div_(logits_temp)
        all_Y = torch.cat(all_Y)

        ens_logits.append(all_logits)

    ens_logits = torch.stack(ens_logits)

    log_p = torch.distributions.Categorical(logits=ens_logits).log_prob(all_Y)
    log_p_bayes = torch.logsumexp(log_p, 0) - torch.log(torch.tensor(log_p.shape[0]))

    gibbs_nll = -log_p.mean()
    bayes_nll = -log_p_bayes.mean()

    Y_pred = ens_logits.softmax(dim=-1).mean(dim=0).argmax(dim=-1)
    acc = (Y_pred == all_Y).sum().item() / Y_pred.size(0)

    probs = torch.nn.functional.softmax(ens_logits, 2).mean(0)
    p, f, w = calibration_curve(probs.cpu().numpy(), all_Y.cpu().numpy())
    ece = expected_calibration_error(p, f, w)

    return {"acc": acc, "gibbs_nll": gibbs_nll, "bayes_nll": bayes_nll, "ece": ece}


def run_sgd(
    train_loader,
    test_loader,
    net,
    criterion,
    device=None,
    lr=1e-2,
    momentum=0.9,
    epochs=1,
):
    train_data = train_loader.dataset
    N = len(train_data)

    sgd = SGD(net.parameters(), lr=lr, momentum=momentum)
    sgd_scheduler = CosineAnnealingLR(sgd, T_max=200)

    best_acc = 0.0

    for e in tqdm(range(epochs)):
        net.train()
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device), Y.to(device)

            sgd.zero_grad()

            f_hat = net(X)
            loss = criterion(f_hat, Y, N=N)

            loss.backward()

            sgd.step()

            if i % 50 == 0:
                metrics = {
                    "epoch": e,
                    "mini_idx": i,
                    "mini_loss": loss.detach().item(),
                }
                wandb.log({f"sgd/train/{k}": v for k, v in metrics.items()}, step=e)

        sgd_scheduler.step()

        test_metrics = test(test_loader, net, criterion, device=device)

        wandb.log({f"sgd/test/{k}": v for k, v in test_metrics.items()}, step=e)

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]

            torch.save(net.state_dict(), Path(wandb.run.dir) / "sgd_model.pt")
            wandb.save("*.pt")
            wandb.run.summary["sgd/test/best_epoch"] = e
            wandb.run.summary["sgd/test/best_acc"] = test_metrics["acc"]

            logging.info(
                f"SGD (Epoch {e}): {wandb.run.summary['sgd/test/best_acc']:.4f}"
            )


def run_sgld(
    train_loader,
    train_loader_eval,
    test_loader,
    net,
    criterion,
    samples_dir,
    device=None,
    lr=1e-2,
    momentum=0.9,
    temperature=1,
    logits_temp=1,
    burn_in=0,
    n_samples=20,
    epochs=1,
    nll_criterion=None,
):
    train_data = train_loader.dataset
    N = len(train_data)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    sgld = SGLD(net.parameters(), lr=lr, momentum=momentum, temperature=temperature)
    sample_int = (epochs - burn_in) // n_samples

    for e in tqdm(range(epochs)):
        net.train()
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device), Y.to(device)

            sgld.zero_grad()

            f_hat = net(X)
            loss = criterion(f_hat, Y, N=N)

            loss.backward()

            sgld.step()

            # if i % 50 == 0:
            #     metrics = {
            #         "epoch": e,
            #         "mini_idx": i,
            #         "mini_loss": loss.detach().item(),
            #     }
            #     wandb.log({f"sgld/train/{k}": v for k, v in metrics.items()}, step=e)

        # test_metrics = test(test_loader, net, criterion, device=device)
        # wandb.log({f"sgld/test/{k}": v for k, v in test_metrics.items()}, step=e)

        # logging.info(f"SGLD (Epoch {e}) : {test_metrics['acc']:.4f}")

        if e + 1 > burn_in and (e + 1 - burn_in) % sample_int == 0:
            torch.save(net.state_dict(), samples_dir / f"s_e{e}.pt")
            wandb.save("samples/*.pt")

            # bma_test_metrics = test_bma(
            #     net,
            #     test_loader,
            #     samples_dir,
            #     nll_criterion=nll_criterion,
            #     device=device,
            # )
            # wandb.log({f"sgld/test/bma_{k}": v for k, v in bma_test_metrics.items()})

            # logging.info(f"SGLD BMA (Epoch {e}): {bma_test_metrics['acc']:.4f}")

            bma_metrics_test = get_metrics_bma(
                net, logits_temp, test_loader, samples_dir, device=device
            )
            wandb.log(
                {f"test/bma_{k}": v for k, v in bma_metrics_test.items()},
                step=e,
            )

            bma_metrics_train = get_metrics_bma(
                net, logits_temp, train_loader_eval, samples_dir, device=device
            )
            wandb.log(
                {f"train/bma_{k}": v for k, v in bma_metrics_train.items()},
                step=e,
            )

        # bma_test_metrics = test_bma(
        #     net, test_loader, samples_dir, nll_criterion=nll_criterion, device=device
        # )
        # wandb.log({f"sgld/test/bma_{k}": v for k, v in bma_test_metrics.items()})
        # wandb.run.summary["sgld/test/bma_acc"] = bma_test_metrics["acc"]

        # logging.info(f"SGLD BMA: {wandb.run.summary['sgld/test/bma_acc']:.4f}")

        mini_loss = loss.detach().item()
        wandb.log({f"train/mini_loss": mini_loss}, step=e)

        log_p_test, acc_test = get_metrics_training(
            net, logits_temp, test_loader, device=device
        )
        wandb.log({f"test/acc": acc_test}, step=e)

        nll_test = -log_p_test.mean().item()
        wandb.log({f"test/nll": nll_test}, step=e)

        params_avg_l2_norm = average_l2_norm(net, num_params)
        wandb.log({f"train/params_avg_l2_norm": params_avg_l2_norm}, step=e)


def run_csgld(
    train_loader,
    train_loader_eval,
    test_loader,
    net,
    criterion,
    samples_dir,
    device=None,
    lr=1e-2,
    momentum=0.9,
    temperature=1,
    logits_temp=1,
    n_samples=20,
    n_cycles=1,
    epochs=1,
    nll_criterion=None,
):
    train_data = train_loader.dataset
    N = len(train_data)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    sgld = SGLD(net.parameters(), lr=lr, momentum=momentum, temperature=temperature)
    sgld_scheduler = CosineLR(
        sgld, n_cycles=n_cycles, n_samples=n_samples, T_max=len(train_loader) * epochs
    )

    for e in tqdm(range(epochs)):
        net.train()
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device), Y.to(device)

            sgld.zero_grad()

            f_hat = net(X)
            loss = criterion(f_hat, Y, N=N)

            loss.backward()

            if sgld_scheduler.get_last_beta() < sgld_scheduler.beta:
                sgld.step(noise=False)
            else:
                sgld.step()

                if sgld_scheduler.should_sample():
                    torch.save(net.state_dict(), samples_dir / f"s_e{e}_m{i}.pt")
                    wandb.save("samples/*.pt")

                    # bma_test_metrics = test_bma(
                    #     net,
                    #     test_loader,
                    #     samples_dir,
                    #     nll_criterion=nll_criterion,
                    #     device=device,
                    # )
                    # wandb.log(
                    #     {f"csgld/test/bma_{k}": v for k, v in bma_test_metrics.items()}
                    # )

                    # logging.info(
                    #     f"cSGLD BMA (Epoch {e}): {bma_test_metrics['acc']:.4f}"
                    # )

                    bma_metrics_test = get_metrics_bma(
                        net, logits_temp, test_loader, samples_dir, device=device
                    )
                    wandb.log(
                        {f"test/bma_{k}": v for k, v in bma_metrics_test.items()},
                        step=e,
                    )

                    bma_metrics_train = get_metrics_bma(
                        net, logits_temp, train_loader_eval, samples_dir, device=device
                    )
                    wandb.log(
                        {f"train/bma_{k}": v for k, v in bma_metrics_train.items()},
                        step=e,
                    )

                    # logging.info(
                    #     f"csgld bma train nll (epoch {e}): {bma_metrics_train['bayes_nll']:.4f}"
                    # )

            sgld_scheduler.step()

            # if i % 50 == 0:
        # metrics = {
        #     "epoch": e,
        #     "mini_loss": loss.detach().item(),
        # }

        mini_loss = loss.detach().item()
        wandb.log({f"train/mini_loss": mini_loss}, step=e)

        # test_metrics = test(test_loader, net, criterion, device=device)

        # wandb.log({f"csgld/test/{k}": v for k, v in test_metrics.items()}, step=e)

        # logging.info(f"cSGLD (Epoch {e}) : {test_metrics['acc']:.4f}")

        log_p_test, acc_test = get_metrics_training(
            net, logits_temp, test_loader, device=device
        )
        wandb.log({f"test/acc": acc_test}, step=e)

        nll_test = -log_p_test.mean().item()
        wandb.log({f"test/nll": nll_test}, step=e)

        params_avg_l2_norm = average_l2_norm(net, num_params)
        wandb.log({f"train/params_avg_l2_norm": params_avg_l2_norm}, step=e)

    # bma_test_metrics = test_bma(
    #     net, test_loader, samples_dir, nll_criterion=nll_criterion, device=device
    # )

    # wandb.log({f"csgld/test/bma_{k}": v for k, v in bma_test_metrics.items()})
    # wandb.run.summary["csgld/test/bma_acc"] = bma_test_metrics["acc"]

    # logging.info(f"cSGLD BMA: {wandb.run.summary['csgld/test/bma_acc']:.4f}")


def main(
    project_name=None,
    wandb_mode=None,
    seed=None,
    device=0,
    data_dir="./",
    ckpt_path=None,
    label_noise=0,
    dataset="cifar10",
    batch_size=128,
    dirty_lik=True,
    prior_scale=1,
    augment=True,
    perm=False,
    noise=0.1,
    likelihood="softmax",
    likelihood_temp=1,
    logits_temp=1,
    epochs=0,
    lr=1e-7,
    sgld_epochs=0,
    sgld_lr=1e-7,
    momentum=0.9,
    temperature=1,
    burn_in=0,
    n_samples=20,
    n_cycles=0,
    test_overfit=False,
):
    if data_dir is None and os.environ.get("DATADIR") is not None:
        data_dir = os.environ.get("DATADIR")
    if ckpt_path:
        ckpt_path = Path(ckpt_path).resolve()

    torch.backends.cudnn.benchmark = True

    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"

    run_name = f"{likelihood_temp}_{logits_temp}_{prior_scale}_{augment}"
    wandb.init(
        project=project_name,
        name=f"{run_name}",
        mode=wandb_mode,
        config={
            "seed": seed,
            "dataset": dataset,
            "batch_size": batch_size,
            "lr": lr,
            "prior_scale": prior_scale,
            "augment": augment,
            "perm": perm,
            "dirty_lik": dirty_lik,
            "temperature": temperature,
            "burn_in": burn_in,
            "sgld_lr": sgld_lr,
            "n_cycles": n_cycles,
            "dir_noise": noise,
            "likelihood": likelihood,
            "likelihood_T": likelihood_temp,
            "logits_temp": logits_temp,
        },
    )

    samples_dir = Path(wandb.run.dir) / "samples"
    samples_dir.mkdir()

    if dataset == "tiny-imagenet":
        train_data, test_data = get_tiny_imagenet(
            root=data_dir, augment=bool(augment), label_noise=label_noise
        )
    elif dataset == "cifar10":
        train_data, test_data = get_cifar10(
            root=data_dir,
            augment=bool(augment),
            label_noise=label_noise,
            perm=bool(perm),
        )
    elif dataset == "cifar100":
        train_data, test_data = get_cifar100(
            root=data_dir,
            augment=bool(augment),
            label_noise=label_noise,
            perm=bool(perm),
        )
    elif dataset == "mnist":
        train_data, test_data = get_mnist(
            root=data_dir,
            augment=bool(augment),
            label_noise=label_noise,
            perm=bool(perm),
        )
    elif dataset == "fmnist":
        train_data, test_data = get_fmnist(
            root=data_dir,
            augment=bool(augment),
            label_noise=label_noise,
            perm=bool(perm),
        )
    else:
        raise NotImplementedError

    if test_overfit:
        train_data = torch.utils.data.Subset(train_data, range(10))
        if dirty_lik == "lenetlarge":
            net = LeNetLarge(num_classes=10).to(device)
        elif dirty_lik == "lenetsmall":
            net = LeNetSmall(num_classes=10).to(device)
        elif dirty_lik == "mlp":
            net = MLP(num_classes=10).to(device)

    if type(augment) is not bool and augment != "true":
        train_data = prepare_transforms(augment=augment, train_data=train_data)
        # train_data.transform = prepare_transforms(augment=augment)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )
    train_loader_eval = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    )

    if not test_overfit:
        if dirty_lik is True or dirty_lik == "resnet18std":
            net = ResNet18(num_classes=train_data.total_classes).to(device)
        elif dirty_lik is False or dirty_lik == "resnet18frn":
            net = ResNet18FRN(num_classes=train_data.total_classes).to(device)
        elif dirty_lik == "resnet18fixup":
            net = ResNet18Fixup(num_classes=train_data.total_classes).to(device)
        elif dirty_lik == "lenetlarge":
            net = LeNetLarge(num_classes=train_data.total_classes).to(device)
        elif dirty_lik == "lenetsmall":
            net = LeNetSmall(num_classes=train_data.total_classes).to(device)
        # print(net)

    net = net.to(device)
    if ckpt_path is not None and ckpt_path.is_file():
        net.load_state_dict(torch.load(ckpt_path))
        logging.info(f"Loaded {ckpt_path}")

    nll_criterion = None
    if likelihood == "dirichlet":
        criterion = KLAugmentedNoisyDirichletLoss(
            net.parameters(),
            num_classes=train_data.total_classes,
            noise=noise,
            likelihood_temp=likelihood_temp,
            prior_scale=prior_scale,
        )
        nll_criterion = NoisyDirichletLoss(
            net.parameters(),
            num_classes=train_data.total_classes,
            noise=noise,
            likelihood_temp=likelihood_temp,
            reduction=None,
        )
    elif likelihood == "softmax":
        criterion = GaussianPriorAugmentedCELoss(
            net.parameters(),
            likelihood_temp=likelihood_temp,
            prior_scale=prior_scale,
            logits_temp=logits_temp,
        )
    else:
        raise NotImplementedError

    if epochs:
        run_sgd(
            train_loader,
            test_loader,
            net,
            criterion,
            device=device,
            lr=lr,
            epochs=epochs,
        )

    if sgld_epochs:
        if n_cycles:
            run_csgld(
                train_loader,
                train_loader_eval,
                test_loader,
                net,
                criterion,
                samples_dir,
                device=device,
                lr=sgld_lr,
                momentum=momentum,
                temperature=temperature,
                n_samples=n_samples,
                n_cycles=n_cycles,
                epochs=sgld_epochs,
                nll_criterion=nll_criterion,
            )
        else:
            run_sgld(
                train_loader,
                train_loader_eval,
                test_loader,
                net,
                criterion,
                samples_dir,
                device=device,
                lr=sgld_lr,
                momentum=momentum,
                temperature=temperature,
                burn_in=burn_in,
                n_samples=n_samples,
                epochs=sgld_epochs,
                nll_criterion=nll_criterion,
            )


if __name__ == "__main__":
    import fire
    import os

    logging.getLogger().setLevel(logging.INFO)

    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="dryrun")
    fire.Fire(main)
