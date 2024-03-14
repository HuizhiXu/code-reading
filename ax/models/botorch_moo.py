#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError
from ax.models.torch.botorch import (
    BotorchModel,
    get_rounding_func,
    TAcqfConstructor,
    TBestPointRecommender,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    recommend_best_observed_point,
    scipy_optimizer,
)
from ax.models.torch.botorch_moo_defaults import (
    get_NEHVI,
    infer_objective_thresholds,
    pareto_frontier_evaluator,
    scipy_optimizer_list,
)
from ax.models.torch.frontier_utils import TFrontierEvaluator
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    _to_inequality_constraints,
    predict_from_model,
    randomize_objective_weights,
    subset_model,
)
from ax.models.torch_base import TorchGenResults, TorchModel, TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from torch import Tensor


logger: Logger = get_logger(__name__)

# pyre-fixme[33]: Aliased annotation cannot contain `Any`.
TOptimizerList = Callable[
    [
        List[AcquisitionFunction],
        Tensor,
        Optional[List[Tuple[Tensor, Tensor, float]]],
        Optional[Dict[int, float]],
        Optional[Callable[[Tensor], Tensor]],
        Any,
    ],
    Tuple[Tensor, Tensor],
]


class MultiObjectiveBotorchModel(BotorchModel):
    r"""
    Customizable multi-objective model.

    默认情况下，它使用期望超体积改进函数来找到具有多个结果的函数的帕累托前沿。通过提供以下组件的自定义实现，可以修改此行为：

    - 一个`model_constructor`，用于在数据上实例化并拟合模型
    - 一个`model_predictor`，使用拟合好的模型进行预测
    - 一个`acqf_constructor`，从拟合好的模型创建一个获取函数
    - 一个`acqf_optimizer`，优化获取函数


    By default, this uses an Expected Hypervolume Improvment function to find the
    pareto frontier of a function with multiple outcomes. This behavior
    can be modified by providing custom implementations of the following
    components:

    - a `model_constructor` that instantiates and fits a model on data
    - a `model_predictor` that predicts outcomes using the fitted model
    - a `acqf_constructor` that creates an acquisition function from a fitted model
    - a `acqf_optimizer` that optimizes the acquisition function

    Args:
        model_constructor: A callable that instantiates and fits a model on data,
            with signature as described below.
        model_predictor: A callable that predicts using the fitted model, with
            signature as described below.
        acqf_constructor: A callable that creates an acquisition function from a
            fitted model, with signature as described below.
        acqf_optimizer: A callable that optimizes an acquisition
            function, with signature as described below.

    "Call signatures"（调用签名）是编程中的一个术语，它描述了函数或方法可以接受的参数的类型、数量和顺序。在Python等动态类型语言中，调用签名有助于开发者理解如何正确地使用一个函数，以及它期望接收什么样的参数。调用签名通常不包括参数的默认值或参数是否可选，这些信息通常在文档字符串（docstrings）或实际的函数定义中给出。

    在Python中，调用签名可以通过装饰器（如@functools.wraps）来复制，或者使用第三方库（如typing模块中的Signature类）来定义。在类型注解中，调用签名可以明确地指出每个参数的类型，以及函数的返回类型。

    Call signatures:

    ::

        model_constructor(
            Xs,
            Ys,
            Yvars,
            task_features,
            fidelity_features,
            metric_names,
            state_dict,
            **kwargs,
        ) -> model

    Here `Xs`, `Ys`, `Yvars` are lists of tensors (one element per outcome),
    `task_features` identifies columns of Xs that should be modeled as a task,
    `fidelity_features` is a list of ints that specify the positions of fidelity
    parameters in 'Xs', `metric_names` provides the names of each `Y` in `Ys`,
    `state_dict` is a pytorch module state dict, and `model` is a BoTorch `Model`.
    Optional kwargs are being passed through from the `BotorchModel` constructor.
    This callable is assumed to return a fitted BoTorch model that has the same
    dtype and lives on the same device as the input tensors.
    s: 这是一个张量列表，每个元素对应一个结果（outcome）。在多目标优化中，每个结果可能代表一个不同的目标函数，例如，在一个同时优化速度和能耗的问题中，速度和能耗就是两个不同的结果）。

    Ys: 这也是一个张量列表，但它包含的是每个结果的目标值。例如，如果Xs中的每个元素代表不同的速度值，那么Ys中对应的元素就代表这些速度值下的性能指标。

    Yvars: 这是一个张量列表，代表Ys中每个结果的目标值的方差。这在统计建模中很重要，因为它允许模型学习数据中的不确定性。

    task_features: 这是一个列标识符，指出Xs中的哪些列应该被视为任务特征。在多任务学习中，某些特征可能对某些任务特别重要。

    fidelity_features: 这是一个整数列表，指定了Xs中保真度参数的位置。保真度参数通常用于控制模型的复杂度或精度，例如，在贝叶斯优化中，可以通过调整噪声水平来控制保真度。

    metric_names: 这是一个字符串列表，为Ys中每个结果提供了名称。这些名称有助于在后续的分析和结果解释中识别不同的目标。

    state_dict: 这是一个PyTorch模块状态字典，包含了模型的参数和状态。这个字典可以用来恢复模型的状态，特别是在加载预训练模型时非常有用。

    **kwargs: 这是一个可变参数，允许BotorchModel构造函数传递额外的关键字参数给model_constructor。这样可以灵活地扩展函数的行为，而不需要修改其签名。

    model: 这是函数返回值，是一个拟合好的BoTorch模型。这个模型应该具有与输入张量相同的数据类型（dtype），并且应该位于与输入张量相同的设备上（例如CPU或GPU）



    ::

        model_predictor(model, X) -> [mean, cov]

    Here `model` is a fitted botorch model, `X` is a tensor of candidate points,
    and `mean` and `cov` are the posterior mean and covariance, respectively.

    ::

        acqf_constructor(
            model,
            objective_weights,
            outcome_constraints,
            X_observed,
            X_pending,
            **kwargs,
        ) -> acq_function


    Here `model` is a botorch `Model`, `objective_weights` is a tensor of weights
    for the model outputs, `outcome_constraints` is a tuple of tensors describing
    the (linear) outcome constraints, `X_observed` are previously observed points,
    and `X_pending` are points whose evaluation is pending. `acq_function` is a
    BoTorch acquisition function crafted from these inputs. For additional
    details on the arguments, see `get_NEI`.

    ::

        acqf_optimizer(
            acq_function,
            bounds,
            n,
            inequality_constraints,
            fixed_features,
            rounding_func,
            **kwargs,
        ) -> candidates

    Here `acq_function` is a BoTorch `AcquisitionFunction`, `bounds` is a tensor
    containing bounds on the parameters, `n` is the number of candidates to be
    generated, `inequality_constraints` are inequality constraints on parameter
    values, `fixed_features` specifies features that should be fixed during
    generation, and `rounding_func` is a callback that rounds an optimization
    result appropriately. `candidates` is a tensor of generated candidates.
    For additional details on the arguments, see `scipy_optimizer`.

    ::

        frontier_evaluator(
            model,
            objective_weights,
            objective_thresholds,
            X,
            Y,
            Yvar,
            outcome_constraints,
        )

    Here `model` is a botorch `Model`, `objective_thresholds` is used in hypervolume
    evaluations, `objective_weights` is a tensor of weights applied to the  objectives
    (sign represents direction), `X`, `Y`, `Yvar` are tensors, `outcome_constraints` is
    a tuple of tensors describing the (linear) outcome constraints.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]
    Xs: List[Tensor]
    Ys: List[Tensor]
    Yvars: List[Tensor]

    def __init__(
        self,
        model_constructor: TModelConstructor = get_and_fit_model,
        model_predictor: TModelPredictor = predict_from_model,
        # pyre-fixme[9]: acqf_constructor has type `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor], Any],
        #  AcquisitionFunction]`; used as `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor],
        #  **(Any)], AcquisitionFunction]`.
        acqf_constructor: TAcqfConstructor = get_NEHVI,
        # pyre-fixme[9]: acqf_optimizer has type `Callable[[AcquisitionFunction,
        #  Tensor, int, Optional[Dict[int, float]], Optional[Callable[[Tensor],
        #  Tensor]], Any], Tensor]`; used as `Callable[[AcquisitionFunction, Tensor,
        #  int, Optional[Dict[int, float]], Optional[Callable[[Tensor], Tensor]],
        #  **(Any)], Tensor]`.
        acqf_optimizer: TOptimizer = scipy_optimizer,
        # TODO: Remove best_point_recommender for botorch_moo. Used in modelbridge._gen.
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        frontier_evaluator: TFrontierEvaluator = pareto_frontier_evaluator,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = False,
        use_input_warping: bool = False,
        use_loocv_pseudo_likelihood: bool = False,
        prior: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_constructor = model_constructor
        self.model_predictor = model_predictor
        self.acqf_constructor = acqf_constructor
        self.acqf_optimizer = acqf_optimizer
        self.best_point_recommender = best_point_recommender
        self.frontier_evaluator = frontier_evaluator
        # pyre-fixme[4]: Attribute must be annotated.
        self._kwargs = kwargs
        self.refit_on_cv = refit_on_cv
        self.refit_on_update = refit_on_update
        self.warm_start_refitting = warm_start_refitting
        self.use_input_warping = use_input_warping
        self.use_loocv_pseudo_likelihood = use_loocv_pseudo_likelihood
        self.prior = prior
        self.model: Optional[Model] = None
        self.Xs = []
        self.Ys = []
        self.Yvars = []
        self.dtype = None
        self.device = None
        self.task_features: List[int] = []
        self.fidelity_features: List[int] = []
        self.metric_names: List[str] = []

    @copy_doc(TorchModel.gen)
    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        options = torch_opt_config.model_gen_options or {}
        acf_options = options.get("acquisition_function_kwargs", {})
        optimizer_options = options.get("optimizer_kwargs", {})

        if search_space_digest.target_fidelities:  # untested
            raise NotImplementedError(
                "target_fidelities not implemented for base BotorchModel"
            )
        if (
            torch_opt_config.objective_thresholds is not None
            and torch_opt_config.objective_weights.shape[0]
            != not_none(torch_opt_config.objective_thresholds).shape[0]
        ):
            raise AxError(
                "Objective weights and thresholds most both contain an element for"
                " each modeled metric."
            )

        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
        )

        model = not_none(self.model)
        full_objective_thresholds = torch_opt_config.objective_thresholds
        full_objective_weights = torch_opt_config.objective_weights
        full_outcome_constraints = torch_opt_config.outcome_constraints
        # subset model only to the outcomes we need for the optimization
        if options.get(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=model,
                objective_weights=torch_opt_config.objective_weights,
                outcome_constraints=torch_opt_config.outcome_constraints,
                objective_thresholds=torch_opt_config.objective_thresholds,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints
            objective_thresholds = subset_model_results.objective_thresholds
            idcs = subset_model_results.indices
        else:
            objective_weights = torch_opt_config.objective_weights
            outcome_constraints = torch_opt_config.outcome_constraints
            objective_thresholds = torch_opt_config.objective_thresholds
            idcs = None
        if (
            objective_thresholds is None
            or objective_thresholds[objective_weights != 0].isnan().any()
        ):
            full_objective_thresholds = infer_objective_thresholds(
                model=model,
                X_observed=not_none(X_observed),
                objective_weights=full_objective_weights,
                outcome_constraints=full_outcome_constraints,
                subset_idcs=idcs,
                objective_thresholds=objective_thresholds,
            )
            # subset the objective thresholds
            objective_thresholds = (
                full_objective_thresholds
                if idcs is None
                else full_objective_thresholds[idcs].clone()
            )

        bounds_ = torch.tensor(
            search_space_digest.bounds, dtype=self.dtype, device=self.device
        )
        bounds_ = bounds_.transpose(0, 1)
        botorch_rounding_func = get_rounding_func(torch_opt_config.rounding_func)
        if acf_options.get("random_scalarization", False) or acf_options.get(
            "chebyshev_scalarization", False
        ):
            # If using a list of acquisition functions, the algorithm to generate
            # that list is configured by acquisition_function_kwargs.
            objective_weights_list = [
                randomize_objective_weights(objective_weights, **acf_options)
                for _ in range(n)
            ]
            acquisition_function_list = [
                self.acqf_constructor(  # pyre-ignore: [28]
                    model=model,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    X_observed=X_observed,
                    X_pending=X_pending,
                    **acf_options,
                )
                for objective_weights in objective_weights_list
            ]
            acquisition_function_list = [
                checked_cast(AcquisitionFunction, acq_function)
                for acq_function in acquisition_function_list
            ]
            # Multiple acquisition functions require a sequential optimizer
            # always use scipy_optimizer_list.
            # TODO(jej): Allow any optimizer.
            candidates, expected_acquisition_value = scipy_optimizer_list(
                acq_function_list=acquisition_function_list,
                bounds=bounds_,
                inequality_constraints=_to_inequality_constraints(
                    linear_constraints=torch_opt_config.linear_constraints
                ),
                fixed_features=torch_opt_config.fixed_features,
                rounding_func=botorch_rounding_func,
                **optimizer_options,
            )
        else:
            acquisition_function = self.acqf_constructor(  # pyre-ignore: [28]
                model=model,
                objective_weights=objective_weights,
                objective_thresholds=objective_thresholds,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                X_pending=X_pending,
                **acf_options,
            )
            acquisition_function = checked_cast(
                AcquisitionFunction, acquisition_function
            )
            # pyre-ignore: [28]
            candidates, expected_acquisition_value = self.acqf_optimizer(
                acq_function=checked_cast(AcquisitionFunction, acquisition_function),
                bounds=bounds_,
                n=n,
                inequality_constraints=_to_inequality_constraints(
                    linear_constraints=torch_opt_config.linear_constraints
                ),
                fixed_features=torch_opt_config.fixed_features,
                rounding_func=botorch_rounding_func,
                **optimizer_options,
            )
        gen_metadata = {
            "expected_acquisition_value": expected_acquisition_value.tolist(),
            "objective_thresholds": not_none(full_objective_thresholds).cpu(),
            "objective_weights": full_objective_weights.cpu(),
        }
        return TorchGenResults(
            points=candidates.detach().cpu(),
            weights=torch.ones(n, dtype=self.dtype),
            gen_metadata=gen_metadata,
        )
