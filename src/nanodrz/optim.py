import math


def warmup_then_linear_decay(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float) -> float:
    if step <= warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > total_steps:
        return min_lr
    else:
        return max_lr - max_lr / (total_steps - warmup_steps) * (step - warmup_steps)


def warmup_then_cosine_decay(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > total_steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)


def warmup_then_inv_sqrt_decay(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    else:
        decay_factor = max_lr * warmup_steps**0.5
        return decay_factor * step**-0.5


def warmup_then_constant(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    else:
        return max_lr
