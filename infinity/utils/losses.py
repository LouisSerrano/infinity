import torch


def per_element_rel_mse_fn(x, y):
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
