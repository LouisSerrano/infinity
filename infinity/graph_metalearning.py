from collections import OrderedDict
from functools import partial

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP


def per_element_rel_mse_fn(x, y, reduction=True):
    num_examples = x.size()[0]

    diff_norms = torch.norm(
        x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1
    )
    y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)

    return diff_norms / y_norms


def batch_mse_rel_fn(x1, x2):
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).
    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).
    Returns:
        MSE tensor of shape (batch_size,).
    """
    # Shape (batch_size, *)
    # per_element_mse = per_element_mse_fn(x1, x2)
    per_element_mse = per_element_rel_mse_fn(x1, x2)
    # Shape (batch_size,)
    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


def inner_loop(
    func_rep,
    modulations,
    coordinates,
    features,
    batch_index,
    n_samples,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
            )
        else:
            fitted_modulations = inner_loop_step(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                batch_index,
                n_samples,
                inner_lr,
                is_train,
                gradient_checkpointing,
            )
    return fitted_modulations


def inner_loop_step(
    func_rep,
    modulations,
    coordinates,
    features,
    batch_index,
    n_samples,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing

    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        features_recon = func_rep.modulated_forward(
            coordinates, modulations[batch_index]
        )
        # loss = element_loss_fn(features_recon, features).mean() * n_samples
        loss = ((features_recon - features) ** 2).mean() * n_samples

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
        # if clip_grad_value is not None:
        #    nn.utils.clip_grad_value_(grad, clip_grad_value)
    # Perform single gradient descent step
    return modulations - inner_lr * grad


def outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    is_train=False,
    return_reconstructions=False,
    gradient_checkpointing=False,
    use_rel_loss=False,
    start_from_zero=True,
    return_dim_loss=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    func_rep.zero_grad()
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    if start_from_zero:
        modulations = torch.zeros_like(graph.modulations).requires_grad_()
    else:
        modulations = graph.modulations.clone().requires_grad_()
    n_samples = len(graph)
    batch_index = graph.batch
    features = graph.images.clone()
    coordinates = graph.pos.clone()

    # Run inner loop
    modulations = inner_loop(
        func_rep,
        modulations,
        coordinates,
        features,
        batch_index,
        n_samples,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(
            coordinates, modulations[batch_index]
        )
        per_example_loss = (features_recon - features) ** 2  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "modulations": modulations,
    }

    if return_reconstructions:
        outputs["reconstructions"] = features_recon

    if use_rel_loss:
        rel_loss = batch_mse_rel_fn(features_recon, features).mean()
        outputs["rel_loss"] = rel_loss
    
    if return_dim_loss:
        dim_loss = per_example_loss.mean(0)
        outputs["dim_loss"] = dim_loss


    return outputs
