import torch


def composite_spectrum_loss(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    if pred.ndim != 2:
        raise ValueError(f"Expected pred and target shape [B, L], got pred={tuple(pred.shape)}")

    peak_weight = 1.0 + 2.0 * target
    weighted_l1 = torch.mean(peak_weight * torch.abs(pred - target))

    grad_pred = pred[:, 1:] - pred[:, :-1]
    grad_target = target[:, 1:] - target[:, :-1]
    grad_l1 = torch.mean(torch.abs(grad_pred - grad_target))

    mse = torch.mean((pred - target) ** 2)
    final_loss = weighted_l1 + 0.1 * grad_l1 + 0.05 * mse

    return {
        "loss": final_loss,
        "weighted_l1": weighted_l1,
        "grad_l1": grad_l1,
        "mse": mse,
    }
