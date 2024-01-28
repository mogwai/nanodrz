import math
import torch


def warmup_then_linear_decay(
    step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float
) -> float:
    if step <= warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > total_steps:
        return min_lr
    else:
        return max_lr - max_lr / (total_steps - warmup_steps) * (step - warmup_steps)


def warmup_then_cosine_decay(
    step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float
):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > total_steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)


def warmup_then_inv_sqrt_decay(
    step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float
):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    else:
        decay_factor = max_lr * warmup_steps**0.5
        return decay_factor * step**-0.5


def warmup_then_constant(
    step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float
):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    else:
        return max_lr


@torch.jit.script
def calculate_smoothed_slope(
    losses: list[float],
    smooth_window_size: int = 3,
    smoothing_constant: float = 0.93,
    regression_win: int = 20,
) -> float:
    y = torch.tensor(losses[-smooth_window_size - 10 :]).view(-1, 1)
    x = torch.arange(y.shape[0]).view(-1, 1)

    # Smooth the data using a simple moving average with a smoothing constant
    smooth_window_size = 3
    conv_weights = (smoothing_constant / smooth_window_size) * torch.ones(
        smooth_window_size
    )
    smoothed_y = torch.nn.functional.conv1d(
        y.view(1, 1, -1),
        conv_weights.view(1, 1, -1),
        padding=(smooth_window_size - 1) // 2,
    ).view(-1)

    regression_win = min(regression_win, smoothed_y.shape[-1])
    # Perform linear regression on the smoothed_y
    x_last_n = x[-regression_win:]
    y_last_n = smoothed_y[-regression_win:]
    X = torch.cat([torch.ones(regression_win, 1), x_last_n], dim=1)
    coefficients = torch.linalg.lstsq(X, y_last_n.view(-1, 1)).solution.flatten()
    return coefficients[1]
