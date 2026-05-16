"""Copula primitives + theta-from-Kendall-tau utilities.

Backport note (lecarb integration): the upstream file uses Python 3.10+
`match`/`case` and `X | Y` type unions. Lecarb runs Python 3.8 so this copy
has been rewritten as if/elif and `Union[X, Y]`. Behavior is unchanged.
"""
from typing import Union

import numpy as np
import torch
from colse.copula_types import CopulaTypes, ArchCopulaTypes
from scipy.stats import kendalltau


copula_type_mapper = {
    CopulaTypes.GUMBEL: ArchCopulaTypes.GUMBEL,
    CopulaTypes.FRANK: ArchCopulaTypes.FRANK,
    CopulaTypes.CLAYTON: ArchCopulaTypes.CLAYTON,
}


def gumbel_copula(u, v, theta):
    part1 = (-np.log(u)) ** theta
    part2 = (-np.log(v)) ** theta
    copula_value = np.exp(-((part1 + part2) ** (1 / theta)))
    return copula_value


def rotated_copula_90(u, v, theta, function):
    return v - function(1 - u, v, theta)


def rotated_copula_180(u, v, theta, function):
    return u + v - 1 + function(1 - u, 1 - v, theta)


def rotated_copula_270(u, v, theta, function):
    return u - function(u, 1 - v, theta)


def get_gumbel_copula(copula_type: ArchCopulaTypes, cdf1, cdf2, theta):
    if copula_type == ArchCopulaTypes.GUMBEL:
        return gumbel_copula(cdf1, cdf2, theta)
    if copula_type == ArchCopulaTypes.GUMBEL_90:
        return rotated_copula_90(cdf1, cdf2, theta, gumbel_copula)
    if copula_type == ArchCopulaTypes.GUMBEL_180:
        return rotated_copula_180(cdf1, cdf2, theta, gumbel_copula)
    if copula_type == ArchCopulaTypes.GUMBEL_270:
        return rotated_copula_270(cdf1, cdf2, theta, gumbel_copula)
    raise ValueError(f"Unhandled gumbel copula_type: {copula_type}")


def get_clayton_copula(copula_type: ArchCopulaTypes, cdf1, cdf2, theta):
    if copula_type == ArchCopulaTypes.CLAYTON:
        return clayton_copula(cdf1, cdf2, theta)
    if copula_type == ArchCopulaTypes.CLAYTON_90:
        return rotated_copula_90(cdf1, cdf2, theta, clayton_copula)
    if copula_type == ArchCopulaTypes.CLAYTON_180:
        return rotated_copula_180(cdf1, cdf2, theta, clayton_copula)
    if copula_type == ArchCopulaTypes.CLAYTON_270:
        return rotated_copula_270(cdf1, cdf2, theta, clayton_copula)
    raise ValueError(f"Unhandled clayton copula_type: {copula_type}")


def gumbel_grad(u, v, theta, copula=None):
    if copula is None:
        copula = get_copula(CopulaTypes.GUMBEL, u, v, theta)
    part_1 = copula * (1 / v) * (-np.log(v)) ** (theta - 1)
    part_2 = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta - 1)
    return part_1 * part_2


def gumbel_copula_torch(u, v, theta):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    epsilon = 1e-5
    u = torch.clamp(u, min=epsilon, max=1 - epsilon)
    v = torch.clamp(v, min=epsilon, max=1 - epsilon)
    part1 = (-torch.log(u)) ** theta
    part2 = (-torch.log(v)) ** theta
    output = -((part1 + part2) ** (1.0 / theta))
    capped_input = torch.clamp(output, max=100.0)
    return torch.exp(capped_input)


def clayton_copula_torch(u, v, theta):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    return torch.maximum(u ** (-theta) + v ** (-theta) - 1, torch.zeros(1)) ** (-1 / theta)


def frank_copula_torch(u, v, theta):
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    part1 = torch.exp(-theta * u) - 1
    part2 = torch.exp(-theta * v) - 1
    part3 = torch.exp(-theta) - 1
    return -1 / theta * torch.log(1 + (part1 * part2) / part3)


def clayton_copula(u, v, theta):
    u = np.asarray(u)
    v = np.asarray(v)
    return np.maximum(u ** (-theta) + v ** (-theta) - 1, 0) ** (-1 / theta)


def frank_copula(u, v, theta):
    u = np.asarray(u)
    v = np.asarray(v)
    part1 = np.exp(-theta * u) - 1
    part2 = np.exp(-theta * v) - 1
    part3 = np.exp(-theta) - 1
    return -1 / theta * np.log(1 + (part1 * part2) / part3)


def get_copula_torch(copula_type: CopulaTypes, cdf1, cdf2, theta, tolerance=1e-5):
    if isinstance(theta, float) or (isinstance(theta, torch.Tensor) and theta.shape == torch.Size([])):
        batch_size = cdf1.shape[0]
        theta = torch.tensor([theta] * batch_size, dtype=torch.float32)
    mask_low = (cdf1 < tolerance) | (cdf2 < tolerance)
    mask_high_both = (cdf1 > (1 - tolerance)) & (cdf2 > (1 - tolerance))
    mask_high_cdf1 = (cdf1 > (1 - tolerance)) & ~(cdf2 > (1 - tolerance))
    mask_high_cdf2 = (cdf2 > (1 - tolerance)) & ~(cdf1 > (1 - tolerance))

    result = torch.where(
        mask_low,
        torch.tensor(0.0, dtype=torch.float32, requires_grad=True),
        torch.zeros_like(cdf1, dtype=torch.float32, requires_grad=True),
    )
    result = torch.where(
        mask_high_both,
        torch.tensor(1.0, dtype=torch.float32, requires_grad=True),
        result,
    )
    result = torch.where(mask_high_cdf1, cdf2.to(torch.float32).requires_grad_(), result)
    result = torch.where(mask_high_cdf2, cdf1.to(torch.float32).requires_grad_(), result)

    remaining_mask = ~(mask_low | mask_high_both | mask_high_cdf1 | mask_high_cdf2)
    copula_result = torch.zeros_like(result, dtype=torch.float32)
    if remaining_mask.any():
        if copula_type == CopulaTypes.CLAYTON:
            copula_values = clayton_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
        elif copula_type == CopulaTypes.GUMBEL:
            copula_values = gumbel_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
        elif copula_type == CopulaTypes.FRANK:
            copula_values = frank_copula_torch(cdf1[remaining_mask], cdf2[remaining_mask], theta[remaining_mask])
        else:
            raise ValueError(f"Unhandled torch copula_type: {copula_type}")
        copula_result[remaining_mask] = copula_values

    return torch.where(remaining_mask, copula_result, result)


def get_copula(copula_type: Union[CopulaTypes, ArchCopulaTypes], cdf1, cdf2, theta, tolerance=1e-5):
    if cdf1 < tolerance or cdf2 < tolerance:
        return 0
    if cdf1 > 1 - tolerance and cdf2 > 1 - tolerance:
        return 1
    elif cdf1 > 1 - tolerance:
        return cdf2
    elif cdf2 > 1 - tolerance:
        return cdf1

    if isinstance(copula_type, CopulaTypes):
        copula_type = copula_type_mapper[copula_type]

    if copula_type.is_gumbel_type():
        return get_gumbel_copula(copula_type, cdf1, cdf2, theta)
    elif copula_type.is_clayton_type():
        return get_clayton_copula(copula_type, cdf1, cdf2, theta)
    else:
        return frank_copula(cdf1, cdf2, theta)


def get_copula_parallel(copula_type, cdf1, cdf2, theta, tolerance=1e-5):
    cdf1 = np.array(cdf1)
    cdf2 = np.array(cdf2)

    result = np.zeros_like(cdf1)
    mask_tolerance = (cdf1 >= tolerance) & (cdf2 >= tolerance)
    mask_upper = (cdf1 > 1 - tolerance) & (cdf2 > 1 - tolerance)
    mask_cdf1_upper = (cdf1 > 1 - tolerance) & ~mask_upper
    mask_cdf2_upper = (cdf2 > 1 - tolerance) & ~mask_upper

    result[mask_upper] = 1
    result[mask_cdf1_upper] = cdf2[mask_cdf1_upper]
    result[mask_cdf2_upper] = cdf1[mask_cdf2_upper]

    mask_compute = mask_tolerance & ~(mask_upper | mask_cdf1_upper | mask_cdf2_upper)
    if isinstance(copula_type, CopulaTypes):
        copula_type = copula_type_mapper[copula_type]
    if copula_type.is_gumbel_type():
        result[mask_compute] = get_gumbel_copula(copula_type, cdf1[mask_compute], cdf2[mask_compute], theta)
    elif copula_type.is_clayton_type():
        result[mask_compute] = get_clayton_copula(copula_type, cdf1[mask_compute], cdf2[mask_compute], theta)
    else:
        result[mask_compute] = frank_copula(cdf1[mask_compute], cdf2[mask_compute], theta)
    return result


# 3.10 match patterns like `case CopulaTypes.CLAYTON | ArchCopulaTypes.CLAYTON:`
# translated to a tuple membership check below.
_CLAYTON_FAMILY = (CopulaTypes.CLAYTON, ArchCopulaTypes.CLAYTON)
_GUMBEL_FAMILY = (
    CopulaTypes.GUMBEL,
    ArchCopulaTypes.GUMBEL,
    ArchCopulaTypes.GUMBEL_90,
    ArchCopulaTypes.GUMBEL_180,
    ArchCopulaTypes.GUMBEL_270,
)
_FRANK_FAMILY = (CopulaTypes.FRANK, ArchCopulaTypes.FRANK)


def get_theta(*args):
    copula_type, data_1, data_2 = args[0]
    if isinstance(copula_type, CopulaTypes):
        copula_type = copula_type_mapper[copula_type]
    if copula_type in _CLAYTON_FAMILY:
        return _get_theta_for_clayton_copula(data_1, data_2)
    if copula_type in _GUMBEL_FAMILY:
        return _get_theta_gumbell(data_1, data_2)
    if copula_type in _FRANK_FAMILY:
        return _get_theta_frank(data_1, data_2)
    raise ValueError(f"Unhandled copula_type for theta: {copula_type}")


def get_theta_from_tau(copula_type: CopulaTypes, tau):
    if copula_type == CopulaTypes.CLAYTON:
        return ((2 * tau) / (1 - tau)) if tau != 1 else 1000
    if copula_type == CopulaTypes.GUMBEL:
        return (1 / (1 - tau)) if tau != 1 else 1000
    raise ValueError(f"Unhandled copula_type for tau: {copula_type}")


def _get_theta_for_clayton_copula(data_1, data_2):
    tau, _ = kendalltau(data_1, data_2)
    theta = ((2 * tau) / (1 - tau)) if tau != 1 else 1000
    return theta


def _get_theta_gumbell(data_1, data_2):
    tau, _ = kendalltau(data_1, data_2)
    if tau is np.nan:
        return False
    theta = (1 / (1 - tau)) if tau != 1 else 1000
    return min(theta, 1000)


def _get_theta_frank(data_1, data_2):
    kendall_tau, _ = kendalltau(data_1, data_2)
    if kendall_tau == 1:
        return float('inf')
    elif kendall_tau == -1:
        return float('-inf')
    else:
        return - (3 * kendall_tau) / (2 * (1 - kendall_tau))


if __name__ == "__main__":
    u = np.array([0.5, 0.7])
    v = np.array([0.5, 0.7])
    theta = np.array([2.0, 1.5])
    tu = torch.tensor(u, dtype=torch.float32)
    tv = torch.tensor(v, dtype=torch.float32)
    theta = torch.tensor(theta, dtype=torch.float32)
    ret = get_copula_torch(CopulaTypes.GUMBEL, tu, tv, theta)
    print(ret)
