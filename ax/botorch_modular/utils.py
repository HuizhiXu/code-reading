#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from contextlib import contextmanager
from logging import Logger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, UnsupportedError
from ax.models.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.models.model import Model, ModelList
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms.input import ChainedInputTransform
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import is_fully_bayesian
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

MIN_OBSERVED_NOISE_LEVEL = 1e-7
logger: Logger = get_logger(__name__)


def use_model_list(
    datasets: List[SupervisedDataset],
    botorch_model_class: Type[Model],
    allow_batched_models: bool = True,
) -> bool:
    """
用于确定是否应该使用模型列表（ModelListGP）来处理数据集
函数的逻辑基于数据集的数量和特性，以及所选择的 BoTorch 模型类。

参数解释：

datasets: 一个包含 SupervisedDataset 对象的列表，每个对象代表一组监督学习数据（特征矩阵 X 和目标向量 Y）。
botorch_model_class: 一个类型参数，表示将要使用的 BoTorch 模型类。
allow_batched_models: 一个布尔值，指示是否允许使用批量模型（默认为 True）。

函数逻辑：

如果 botorch_model_class 是 MultiTaskGP 的子类，函数返回 True，表示应该使用模型列表，因为多任务模型总是被包装在 ModelListGP 中。

如果 botorch_model_class 是 SaasFullyBayesianSingleTaskGP 的子类，函数检查是否有多个数据集或者第一个数据集的 Y 有多个输出。如果是这样，函数返回 True，表示应该使用模型列表，因为 SAAS 模型不支持多个输出。

如果只有一个数据集，函数返回 False，表示可以使用单一模型，而不是模型列表。

如果 botorch_model_class 是 BatchedMultiOutputGPyTorchModel 的子类，并且所有数据集的 X 都是相等的，函数检查 allow_batched_models 参数。如果不允许批量模型，函数返回 True，表示应该使用模型列表。

如果有多个不同的 X，或者它们不全都相等，函数返回 True，表示应该使用 ListSurrogate 和 ModelListGP，因为这种情况下无法使用单一模型或批量模型。

总结来说，这个函数的目的是确定在处理多任务、多输出或多个数据集时，是否应该使用 BoTorch 的模型列表功能。这取决于数据集的特性和所选择的模型类。

    
    """

    if issubclass(botorch_model_class, MultiTaskGP):
        # We currently always wrap multi-task models into `ModelListGP`.
        return True
    elif issubclass(botorch_model_class, SaasFullyBayesianSingleTaskGP):
        # SAAS models do not support multiple outcomes.
        # Use model list if there are multiple outcomes.
        return len(datasets) > 1 or datasets[0].Y().shape[-1] > 1
    elif len(datasets) == 1:
        # Just one outcome, can use single model.
        return False
    elif issubclass(botorch_model_class, BatchedMultiOutputGPyTorchModel) and all(
        torch.equal(datasets[0].X(), ds.X()) for ds in datasets[1:]
    ):
        # Use batch models if allowed
        return not allow_batched_models
    # If there are multiple Xs and they are not all equal, we
    # use `ListSurrogate` and `ModelListGP`.
    return True


def choose_model_class(
    datasets: List[SupervisedDataset],
    search_space_digest: SearchSpaceDigest,
) -> Type[Model]:
    """Chooses a BoTorch `Model` using the given data (currently just Yvars)
    and its properties (information about task and fidelity features).

    Args:
        Yvars: List of tensors, each representing observation noise for a
            given outcome, where outcomes are in the same order as in Xs.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A BoTorch `Model` class.
    """
    if len(search_space_digest.fidelity_features) > 1:
        raise NotImplementedError(
            "Only a single fidelity feature supported "
            f"(got: {search_space_digest.fidelity_features})."
        )
    if len(search_space_digest.task_features) > 1:
        raise NotImplementedError(
            f"Only a single task feature supported "
            f"(got: {search_space_digest.task_features})."
        )
    if search_space_digest.task_features and search_space_digest.fidelity_features:
        raise NotImplementedError(
            "Multi-task multi-fidelity optimization not yet supported."
        )

    is_fixed_noise = [ds.Yvar is not None for ds in datasets]
    all_inferred = not any(is_fixed_noise)
    if not all_inferred and not all(is_fixed_noise):
        raise ValueError(
            "Mix of known and unknown variances indicates valuation function "
            "errors. Variances should all be specified, or none should be."
        )

    # Multi-task cases (when `task_features` specified).
    if search_space_digest.task_features and all_inferred:
        model_class = MultiTaskGP  # Unknown observation noise.
    elif search_space_digest.task_features:
        model_class = FixedNoiseMultiTaskGP  # Known observation noise.

    # Single-task multi-fidelity cases.
    elif search_space_digest.fidelity_features and all_inferred:
        model_class = SingleTaskMultiFidelityGP  # Unknown observation noise.
    elif search_space_digest.fidelity_features:
        model_class = FixedNoiseMultiFidelityGP  # Known observation noise.

    # Mixed optimization case. Note that presence of categorical
    # features in search space digest indicates that downstream in the
    # stack we chose not to perform continuous relaxation on those
    # features.
    elif search_space_digest.categorical_features:
        if not all_inferred:
            logger.warning(
                "Using `MixedSingleTaskGP` despire the known `Yvar` values. This "
                "is a temporary measure while fixed-noise mixed BO is in the works."
            )
        model_class = MixedSingleTaskGP

    # Single-task single-fidelity cases.
    elif all_inferred:  # Unknown observation noise.
        model_class = SingleTaskGP
    else:
        model_class = FixedNoiseGP  # Known observation noise.

    logger.debug(f"Chose BoTorch model class: {model_class}.")
    return model_class


def choose_botorch_acqf_class(
    pending_observations: Optional[List[Tensor]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    objective_thresholds: Optional[Tensor] = None,
    objective_weights: Optional[Tensor] = None,
) -> Type[AcquisitionFunction]:
    """Chooses a BoTorch `AcquisitionFunction` class."""
    if objective_thresholds is not None or (
        # using objective_weights is a less-than-ideal fix given its ambiguity,
        # the real fix would be to revisit the infomration passed down via
        # the modelbridge (and be explicit about whether we scalarize or perform MOO)
        objective_weights is not None
        and objective_weights.nonzero().numel() > 1
    ):
        acqf_class = qNoisyExpectedHypervolumeImprovement
    else:
        acqf_class = qLogNoisyExpectedImprovement

    logger.debug(f"Chose BoTorch acquisition function class: {acqf_class}.")
    return acqf_class


def construct_acquisition_and_optimizer_options(
    acqf_options: TConfig, model_gen_options: Optional[TConfig] = None
) -> Tuple[TConfig, TConfig]:
    """Extract acquisition and optimizer options from `model_gen_options`."""
    acq_options = acqf_options.copy()
    opt_options = {}

    if model_gen_options:
        acq_options.update(
            checked_cast(dict, model_gen_options.get(Keys.ACQF_KWARGS, {}))
        )
        # TODO: Add this if all acq. functions accept the `subset_model`
        # kwarg or opt for kwarg filtering.
        # acq_options[SUBSET_MODEL] = model_gen_options.get(SUBSET_MODEL)
        opt_options = checked_cast(
            dict, model_gen_options.get(Keys.OPTIMIZER_KWARGS, {})
        ).copy()
    return acq_options, opt_options


def convert_to_block_design(
    datasets: List[SupervisedDataset],
    metric_names: List[str],
    force: bool = False,
) -> Tuple[List[SupervisedDataset], List[str]]:
    # Convert data to "block design". TODO: Figure out a better
    # solution for this using the data containers (pass outcome
    # names as properties of the data containers)
    is_fixed = [ds.Yvar is not None for ds in datasets]
    if any(is_fixed) and not all(is_fixed):
        raise UnsupportedError(
            "Cannot convert mixed data with and without variance "
            "observaitons to `block design`."
        )
    is_fixed = all(is_fixed)
    Xs = [dataset.X() for dataset in datasets]
    metric_names = ["_".join(metric_names)]  # TODO: Improve this.

    if len({X.shape for X in Xs}) != 1 or not all(
        torch.equal(X, Xs[0]) for X in Xs[1:]
    ):
        if not force:
            raise UnsupportedError(
                "Cannot convert data to non-block design data. "
                "To force this and drop data not shared between "
                "outcomes use `force=True`."
            )
        warnings.warn(
            "Forcing converion of data not complying to a block design "
            "to block design by dropping observations that are not shared "
            "between outcomes.",
            AxWarning,
        )
        X_shared, idcs_shared = _get_shared_rows(Xs=Xs)
        Y = torch.cat([ds.Y()[i] for ds, i in zip(datasets, idcs_shared)], dim=-1)
        if is_fixed:
            Yvar = torch.cat(
                [ds.Yvar()[i] for ds, i in zip(datasets, idcs_shared)],
                dim=-1,
            )
            datasets = [SupervisedDataset(X=X_shared, Y=Y, Yvar=Yvar)]
        else:
            datasets = [SupervisedDataset(X=X_shared, Y=Y)]
        return datasets, metric_names

    # data complies to block design, can concat with impunity
    Y = torch.cat([ds.Y() for ds in datasets], dim=-1)
    if is_fixed:
        Yvar = torch.cat([not_none(ds.Yvar)() for ds in datasets], dim=-1)
        datasets = [SupervisedDataset(X=Xs[0], Y=Y, Yvar=Yvar)]
    else:
        datasets = [SupervisedDataset(X=Xs[0], Y=Y)]
    return datasets, metric_names


def _get_shared_rows(Xs: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    """Extract shared rows from a list of tensors

    Args:
        Xs: A list of m two-dimensional tensors with shapes
            `(n_1 x d), ..., (n_m x d)`. It is not required that
            the `n_i` are the same.

    Returns:
        A two-tuple containing (i) a Tensor with the rows that are
        shared between all the Tensors in `Xs`, and (ii) a list of
        index tensors that indicate the location of these rows
        in the respective elements of `Xs`.
    """
    idcs_shared = []
    Xs_sorted = sorted(Xs, key=len)
    X_shared = Xs_sorted[0].clone()
    for X in Xs_sorted[1:]:
        X_shared = X_shared[(X_shared == X.unsqueeze(-2)).all(dim=-1).any(dim=-2)]
    # get indices
    for X in Xs:
        same = (X_shared == X.unsqueeze(-2)).all(dim=-1).any(dim=-1)
        idcs_shared.append(torch.arange(same.shape[-1], device=X_shared.device)[same])
    return X_shared, idcs_shared


def fit_botorch_model(
    model: Model,
    mll_class: Type[MarginalLogLikelihood],
    mll_options: Optional[Dict[str, Any]] = None,
) -> None:
    """Fit a BoTorch model."""
    mll_options = mll_options or {}
    models = model.models if isinstance(model, ModelList) else [model]
    for m in models:
        # TODO: Support deterministic models when we support `ModelList`
        if is_fully_bayesian(m):
            fit_fully_bayesian_model_nuts(
                m,
                disable_progbar=True,
                jit_compile=mll_options.get("jit_compile", False),
            )
        elif isinstance(m, (GPyTorchModel, PairwiseGP)):
            mll_options = mll_options or {}
            mll = mll_class(likelihood=m.likelihood, model=m, **mll_options)
            fit_gpytorch_mll(mll)
        else:
            raise NotImplementedError(
                f"Model of type {m.__class__.__name__} is currently not supported."
            )


@contextmanager
def disable_one_to_many_transforms(model: Model) -> Generator[None, None, None]:
    r"""A context manager for temporarily disabling one-to-many transforms.

    This can be used to avoid perturbing the user supplied inputs when
    getting the predictions from the model.

    NOTE: This currently does not support chained input transforms.

    Args:
        model: The BoTorch `Model` to disable the transforms for.
    """
    models = model.models if isinstance(model, ModelList) else [model]
    input_transforms = [getattr(m, "input_transform", None) for m in models]
    try:
        for intf in input_transforms:
            if intf is None:
                continue
            if isinstance(intf, ChainedInputTransform):
                raise UnsupportedError(
                    "ChainedInputTransforms are currently not supported."
                )
            if intf.is_one_to_many:
                intf.transform_on_eval = False
        yield
    finally:
        for intf in input_transforms:
            if intf is not None and intf.is_one_to_many:
                intf.transform_on_eval = True


def _tensor_difference(A: Tensor, B: Tensor) -> Tensor:
    """Used to return B sans any Xs that also appear in A"""
    C = torch.cat((A, B), dim=0)
    D, inverse_ind = torch.unique(C, return_inverse=True, dim=0)
    n = A.shape[0]
    A_indices = inverse_ind[:n].tolist()
    B_indices = inverse_ind[n:].tolist()
    Bi_set = set(B_indices) - set(A_indices)
    return D[list(Bi_set)]


def get_post_processing_func(
    rounding_func: Optional[Callable[[Tensor], Tensor]],
    optimizer_options: Dict[str, Any],
) -> Optional[Callable[[Tensor], Tensor]]:
    """Get the post processing function by combining the rounding function
    with the post processing function provided as part of the optimizer
    options. If both are given, the post processing function is applied before
    applying the rounding function. If only one of them is given, then
    it is used as the post processing function.
    """
    if "post_processing_func" in optimizer_options:
        provided_func: Callable[[Tensor], Tensor] = optimizer_options.pop(
            "post_processing_func"
        )
        if rounding_func is None:
            # No rounding function is given. We can use the post processing
            # function directly.
            return provided_func
        else:
            # Both post processing and rounding functions are given. We need
            # to chain them and apply the post processing function first.
            base_rounding_func: Callable[[Tensor], Tensor] = rounding_func

            def combined_func(x: Tensor) -> Tensor:
                return base_rounding_func(provided_func(x))

            return combined_func

    else:
        return rounding_func
