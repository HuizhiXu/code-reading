#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TGenMetadata
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.models.torch.botorch import (
    get_feature_importances_from_botorch_model,
    get_rounding_func,
)
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    construct_acquisition_and_optimizer_options,
    convert_to_block_design,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchGenResults, TorchModel, TorchOptConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelList
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.datasets import SupervisedDataset
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

T = TypeVar("T")


def single_surrogate_only(f: Callable[..., T]) -> Callable[..., T]:
    """
    包装那些只有单个代理模型（Surrogate）的 BoTorchModel 类的方法的装饰器。
    """

    @wraps(f)
    def impl(self: "BoTorchModel", *args: List[Any], **kwargs: Dict[str, Any]) -> T:
        if len(self._surrogates) != 1:
            raise NotImplementedError(
                f"{f.__name__} not implemented for multi-surrogate case. Found "
                f"{self.surrogates=}."
            )
        return f(self, *args, **kwargs)

    return impl

# @dataclass(frozen=True)：这是一个装饰器，用于自动为类生成特殊方法（如 __init__、__repr__ 等），并使类不可变（immutable）。frozen=True 表示一旦创建了类的实例，其属性就不能被修改。
@dataclass(frozen=True)
class SurrogateSpec:
    """
    定义了一个名为 SurrogateSpec 的数据类，提供了一种结构化的方式来指定代理模型的参数和设置。
    当调用 BotorchModel.fit 方法时，SurrogateSpec 类的字段将被用来构建所需的代理模型对象。
    如果 outcomes 列表为空，则不会将任何输出变量拟合到代理模型中。

    botorch_model_class：代理模型的 BoTorch 类型，它是 Model 类的子类。
    mll_class：边际对数似然（Marginal LogLikelihood）的类类型，用于评估代理模型的质量。默认为 ExactMarginalLogLikelihood。
    covar_module_class：协方差模块（Covariance Module）的类类型，它是 Kernel 类的子类。默认值为 None，表示没有指定特定的协方差模块类。
    likelihood_class：似然函数（Likelihood Function）的类类型，用于定义观测数据的概率分布。默认值为 None。

    """

    botorch_model_class: Optional[Type[Model]] = None
    botorch_model_kwargs: Dict[str, Any] = field(default_factory=dict)

    mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood
    mll_kwargs: Dict[str, Any] = field(default_factory=dict)

    covar_module_class: Optional[Type[Kernel]] = None
    covar_module_kwargs: Optional[Dict[str, Any]] = None

    likelihood_class: Optional[Type[Likelihood]] = None
    likelihood_kwargs: Optional[Dict[str, Any]] = None

    input_transform: Optional[InputTransform] = None
    outcome_transform: Optional[OutcomeTransform] = None

    allow_batched_models: bool = True

    outcomes: List[str] = field(default_factory=list)


class BoTorchModel(TorchModel, Base):
    """
    Ax把BoTorch 的 `Model` 和 `AcquisitionFunction`封装成ax中的`Model`类里面的 `Surrogate` 和 `Acquisition` 对象。
    BoTorchModel 是Model里的一种。

    BoTorchModel 类提供了一种灵活的方式来配置和使用 BoTorch 子组件进行贝叶斯优化。
    它允许用户通过 SurrogateSpec 来自定义代理模型的初始化，或者直接提供一个 Surrogate 实例。
    此外，它还提供了对获取函数（Acquisition Function）的配置选项，以及在模型更新和交叉验证时重新优化模型参数的选项。


     Args:
    acquisition_class：用于此模型的 Acquisition 类型
    acquisition_options：传递给 BoTorch AcquisitionFunction 构造函数的可选关键字参数字典。
    botorch_acqf_class：用于此模型的 AcquisitionFunction 类型
    surrogate_specs：指定如何初始化特定代理模型以模拟特定输出的名称到 SurrogateSpec 的可选映射。
    作为 SurrogateSpecs 的替代，可以提供一个 Surrogate 实例，用作所有输出的唯一代理模型。
    refit_on_update：在调用 BoTorchModel.update 时是否重新优化模型参数的布尔值。


    Args:
        acquisition_class: Type of `Acquisition` to be used in
            this model, auto-selected based on experiment and data
            if not specified.
        acquisition_options: Optional dict of kwargs, passed to
            the constructor of BoTorch `AcquisitionFunction`.
        botorch_acqf_class: Type of `AcquisitionFunction` to be
            used in this model, auto-selected based on experiment
            and data if not specified.
        surrogate_specs: Optional Mapping of names onto SurrogateSpecs, which specify
            how to initialize specific Surrogates to model specific outcomes. If None
            is provided a single Surrogate will be created and set up automatically
            based on the data provided.
        surrogate: In liu of SurrogateSpecs, an instance of `Surrogate` may be
            provided to be used as the sole Surrogate for all outcomes
        refit_on_update: Whether to reoptimize model parameters during call
            to `BoTorchModel.update`. If false, training data for the model
            (used for inference) is still swapped for new training data, but
            model parameters are not reoptimized.
        refit_on_cv: Whether to reoptimize model parameters during call to
            `BoTorchmodel.cross_validate`.
        warm_start_refit: Whether to load parameters from either the provided
            state dict or the state dict of the current BoTorch `Model` during
            refitting. If False, model parameters will be reoptimized from
            scratch on refit. NOTE: This setting is ignored during `update` or
            `cross_validate` if the corresponding `refit_on_...` is False.
    """

    acquisition_class: Type[Acquisition]
    acquisition_options: Dict[str, Any]

    surrogate_specs: Dict[str, SurrogateSpec]
    _surrogates: Dict[str, Surrogate]

    _botorch_acqf_class: Optional[Type[AcquisitionFunction]]
    _search_space_digest: Optional[SearchSpaceDigest] = None
    _supports_robust_optimization: bool = True

    def __init__(
        self,
        surrogate_specs: Optional[Mapping[str, SurrogateSpec]] = None,
        surrogate: Optional[Surrogate] = None,
        acquisition_class: Optional[Type[Acquisition]] = None,
        acquisition_options: Optional[Dict[str, Any]] = None,
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        refit_on_update: bool = True,
        refit_on_cv: bool = False,
        warm_start_refit: bool = True,
    ) -> None:
        """
        
        确保只提供了 surrogate_specs 或 surrogate 中的一个，而不是两者都提供。
        确保在 SurrogateSpecs 中，每个输出只由一个代理模型建模。
        确保用户没有使用保留的代理模型标签。
        初始化 _surrogates 字典和 surrogate_specs 字典。
        设置 acquisition_class、acquisition_options 和 _botorch_acqf_class。
        设置 refit_on_update、refit_on_cv 和 warm_start_refit。
        """
        # Ensure only surrogate_specs or surrogate is provided
        if surrogate_specs and surrogate:
            raise UserInputError(
                "Only one of `surrogate_specs` and `surrogate` arguments is expected."
            )

        # Ensure each outcome is only modeled by one Surrogate in the SurrogateSpecs
        if surrogate_specs is not None:
            outcomes_by_surrogate_label = {
                label: spec.outcomes for label, spec in surrogate_specs.items()
            }
            if sum(
                len(outcomes) for outcomes in outcomes_by_surrogate_label.values()
            ) != len(set(*outcomes_by_surrogate_label.values())):
                raise UserInputError(
                    "Each outcome may be modeled by only one Surrogate, found "
                    f"{outcomes_by_surrogate_label}"
                )

        # Ensure user does not use reserved Surrogate labels
        if (
            surrogate_specs is not None
            and len(
                {Keys.ONLY_SURROGATE, Keys.AUTOSET_SURROGATE} - surrogate_specs.keys()
            )
            < 2
        ):
            raise UserInputError(
                f"SurrogateSpecs may not be labeled {Keys.ONLY_SURROGATE} or "
                f"{Keys.AUTOSET_SURROGATE}, these are reserved."
            )

        self._surrogates = {}
        self.surrogate_specs = {}
        if surrogate_specs is not None:
            self.surrogate_specs: Dict[str, SurrogateSpec] = {
                label: spec for label, spec in surrogate_specs.items()
            }
        elif surrogate is not None:
            self._surrogates = {Keys.ONLY_SURROGATE: surrogate}

        self.acquisition_class = acquisition_class or Acquisition
        self.acquisition_options = acquisition_options or {}
        self._botorch_acqf_class = botorch_acqf_class

        self.refit_on_update = refit_on_update
        self.refit_on_cv = refit_on_cv
        self.warm_start_refit = warm_start_refit

    @property
    def surrogates(self) -> Dict[str, Surrogate]:
        """Surrogates by label"""
        return self._surrogates

    @property
    @single_surrogate_only
    def surrogate(self) -> Surrogate:
        """Surrogate, if there is only one."""

        return next(iter(self.surrogates.values()))

    @property
    @single_surrogate_only
    def Xs(self) -> List[Tensor]:
        """A list of tensors, each of shape ``batch_shape x n_i x d``,
        where `n_i` is the number of training inputs for the i-th model.

        NOTE: This is an accessor for ``self.surrogate.Xs``
        and returns it unchanged.
        """

        return self.surrogate.Xs

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class, associated with this model.
        Raises an error if one is not yet set.
        """
        if not self._botorch_acqf_class:
            raise ValueError("BoTorch `AcquisitionFunction` has not yet been set.")
        return self._botorch_acqf_class

    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        # state dict by surrogate label
        state_dicts: Optional[Mapping[str, Dict[str, Tensor]]] = None,
        refit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Fit model to m outcomes.
        fit 方法是 BoTorchModel 类中的一个核心方法，它负责根据提供的数据集和指标名称训练模型。
        这个方法处理了多个代理模型的拟合过程，包括预先构建的代理模型和根据 SurrogateSpec 创建的新代理模型。

        Args:
        datasets: 一个 SupervisedDataset 容器的列表，每个容器对应一个指标（输出）的数据。
        metric_names: 一个指标名称的列表，第 i 个指标对应于第 i 个数据集。
        search_space_digest: 一个 SearchSpaceDigest 对象，包含数据集中特征的元数据信息。
        candidate_metadata: 可选参数，模型为候选点生成的元数据列表，顺序与 Xs 对应。
        state_dicts: 可选参数，按模型标签加载的状态字典，通过 surrogate_specs 传递。
        refit: 一个布尔值，指示是否重新优化模型参数。


        细节：

        首先，方法检查 datasets 和 metric_names 的长度是否匹配，如果不匹配，则抛出 ValueError 异常。

        接着，方法存储 search_space_digest 以供后续使用（例如在生成过程中）。

        如果用户传入了一个预先构建的代理模型，并且 _surrogates 字典中包含 Keys.ONLY_SURROGATE 键，则假定我们要拟合所有指标。在这种情况下，方法将调用该代理模型的 fit 方法，并返回。

        如果没有预先构建的代理模型，方法将根据 surrogate_specs 初始化一个 Surrogate 对每个指标。对于每个 SurrogateSpec，都会创建一个新的 Surrogate 实例。

        如果有任何未明确分配给代理模型的指标，且 surrogates 的数量不为 1，则为这些指标创建一个新的 Surrogate（这将自动为每个指标设置其 BoTorch 模型类）。

        然后，方法将数据集按指标名称分组，并为每个 Surrogate 迭代地调用其 fit 方法，传入分配给该代理模型的指标的数据集和名称。

        如果 Surrogate 的模型不是 ModelList 且数据集不符合块设计（block design），则调用 convert_to_block_design 函数来转换数据集，确保数据符合块设计。

        最后，使用 subset_datasets 和 subset_metric_names 调用每个 Surrogate 的 fit 方法，传入状态字典、是否重新优化模型参数以及其他关键字参数
                
        """

        if len(datasets) != len(metric_names):
            raise ValueError(
                "Length of datasets and metric_names must match, but your inputs "
                f"are of lengths {len(datasets)} and {len(metric_names)}, "
                "respectively."
            )

        # Store search space info for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        # Step 0. If the user passed in a preconstructed surrogate we won't have a
        # SurrogateSpec and must assume we're fitting all metrics
        if Keys.ONLY_SURROGATE in self._surrogates.keys():
            self._surrogates[Keys.ONLY_SURROGATE].fit(
                datasets=datasets,
                metric_names=metric_names,
                search_space_digest=search_space_digest,
                candidate_metadata=candidate_metadata,
                state_dict=state_dicts.get(Keys.ONLY_SURROGATE)
                if state_dicts
                else None,
                refit=refit,
                **kwargs,
            )
            return

        # Step 1. Initialize a Surrogate for every SurrogateSpec
        self._surrogates = {
            label: Surrogate(
                # if None, Surrogate will autoset class per outcome at construct time
                botorch_model_class=spec.botorch_model_class,
                model_options=spec.botorch_model_kwargs,
                mll_class=spec.mll_class,
                mll_options=spec.mll_kwargs,
                covar_module_class=spec.covar_module_class,
                covar_module_options=spec.covar_module_kwargs,
                likelihood_class=spec.likelihood_class,
                likelihood_options=spec.likelihood_kwargs,
                input_transform=spec.input_transform,
                outcome_transform=spec.outcome_transform,
                allow_batched_models=spec.allow_batched_models,
            )
            for label, spec in self.surrogate_specs.items()
        }

        # Step 1.5. If any outcomes are not explicitly assigned to a Surrogate, create
        # a new Surrogate for all these metrics (which will autoset its botorch model
        # class per outcome) UNLESS there is only one SurrogateSpec with no outcomes
        # assigned to it, in which case that will be used for all metrics.
        assigned_metric_names = {
            item
            for sublist in [spec.outcomes for spec in self.surrogate_specs.values()]
            for item in sublist
        }
        unassigned_metric_names = [
            name for name in metric_names if name not in assigned_metric_names
        ]
        if len(unassigned_metric_names) > 0 and len(self.surrogates) != 1:
            self._surrogates[Keys.AUTOSET_SURROGATE] = Surrogate()

        # Step 2. Fit each Surrogate iteratively using its assigned outcomes
        datasets_by_metric_name = dict(zip(metric_names, datasets))
        for label, surrogate in self.surrogates.items():
            if label == Keys.AUTOSET_SURROGATE or len(self.surrogates) == 1:
                subset_metric_names = unassigned_metric_names
            else:
                subset_metric_names = self.surrogate_specs[label].outcomes

            subset_datasets = [
                datasets_by_metric_name[metric_name]
                for metric_name in subset_metric_names
            ]
            if (
                len(subset_datasets) > 1
                # if Surrogate's model is none a ModelList will be autoset
                and surrogate._model is not None
                and not isinstance(surrogate.model, ModelList)
            ):
                # Note: If the datasets do not confirm to a block design then this
                # will filter the data and drop observations to make sure that it does.
                # This can happen e.g. if only some metrics are observed at some points
                subset_datasets, metric_names = convert_to_block_design(
                    datasets=subset_datasets,
                    metric_names=metric_names,
                    force=True,
                )

            surrogate.fit(
                datasets=subset_datasets,
                metric_names=subset_metric_names,
                search_space_digest=search_space_digest,
                candidate_metadata=candidate_metadata,
                state_dict=(state_dicts or {}).get(label),
                refit=refit,
                **kwargs,
            )

    @copy_doc(TorchModel.update)
    def update(
        self,
        datasets: List[Optional[SupervisedDataset]],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        用于更新模型以包含新的数据集。它首先检查模型是否已经拟合过（即是否有代理模型存在），如果没有，则抛出 UnsupportedError 异常。
        然后，它存储搜索空间摘要信息以供后续使用，并迭代每个代理模型，根据分配的指标更新它们。
        """
        if len(self.surrogates) == 0:
            raise UnsupportedError("Cannot update model that has not been fitted.")

        # store search space info  for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        for label, surrogate in self.surrogates.items():
            # Sometimes the model fit should be restarted from scratch on update, for
            # models that are prone to overfitting. In those cases,
            # `self.warm_start_refit` should be false and `Surrogate.update` will not
            # receive a state dict and will not pass it to the underlying
            # `Surrogate.fit`.

            state_dict = (
                None
                if self.refit_on_update and not self.warm_start_refit
                else surrogate.model.state_dict()
            )
            if any(dataset is None for dataset in datasets):
                raise UnsupportedError(
                    f"{self.__class__.__name__}.update requires data for all outcomes."
                )

            # Only update each Surrogate on its own metrics unless it was the
            # preconstructed Surrogate
            datasets_by_metric_name = dict(zip(metric_names, datasets))
            subset_metric_names = (
                self.surrogate_specs[label].outcomes
                if label not in (Keys.ONLY_SURROGATE, Keys.AUTOSET_SURROGATE)
                else metric_names
            )
            subset_datasets = [
                datasets_by_metric_name[metric_name]
                for metric_name in subset_metric_names
            ]

            surrogate.update(
                datasets=subset_datasets,
                metric_names=subset_metric_names,
                search_space_digest=search_space_digest,
                candidate_metadata=candidate_metadata,
                state_dict=state_dict,
                refit=self.refit_on_update,
                **kwargs,
            )

    @single_surrogate_only
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict if only one Surrogate, Error if there are many
        如果模型只有一个代理模型，则此方法允许进行预测。如果有多个代理模型，则抛出错误。
        """
        
        return self.surrogate.predict(X=X)

    def predict_from_surrogate(
        self, surrogate_label: str, X: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Predict from the Surrogate with the given label.允许从具有给定标签的特定代理模型进行预测。"""
        return self.surrogates[surrogate_label].predict(X=X)

    @copy_doc(TorchModel.gen)
    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        """
        生成新的候选点。它构建获取函数和优化器选项，更新搜索空间摘要，实例化获取函数，并执行优化以生成新的候选点。

        """
        acq_options, opt_options = construct_acquisition_and_optimizer_options(
            acqf_options=self.acquisition_options,
            model_gen_options=torch_opt_config.model_gen_options,
        )
        # update bounds / target fidelities
        search_space_digest = dataclasses.replace(
            self.search_space_digest,
            bounds=search_space_digest.bounds,
            target_fidelities=search_space_digest.target_fidelities or {},
        )

        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        botorch_rounding_func = get_rounding_func(torch_opt_config.rounding_func)
        candidates, expected_acquisition_value = acqf.optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=torch_opt_config.linear_constraints
            ),
            fixed_features=torch_opt_config.fixed_features,
            rounding_func=botorch_rounding_func,
            optimizer_options=checked_cast(dict, opt_options),
        )
        gen_metadata = self._get_gen_metadata_from_acqf(
            acqf=acqf,
            torch_opt_config=torch_opt_config,
            expected_acquisition_value=expected_acquisition_value,
        )
        return TorchGenResults(
            points=candidates.detach().cpu(),
            weights=torch.ones(n, dtype=self.dtype),
            gen_metadata=gen_metadata,
        )

    def _get_gen_metadata_from_acqf(
        self,
        acqf: Acquisition,
        torch_opt_config: TorchOptConfig,
        expected_acquisition_value: Tensor,
    ) -> TGenMetadata:
        gen_metadata: TGenMetadata = {
            Keys.EXPECTED_ACQF_VAL: expected_acquisition_value.tolist()
        }
        if torch_opt_config.objective_weights.nonzero().numel() > 1:
            gen_metadata["objective_thresholds"] = acqf.objective_thresholds
            gen_metadata["objective_weights"] = acqf.objective_weights

        if hasattr(acqf.acqf, "outcome_model"):
            outcome_model = acqf.acqf.outcome_model
            if isinstance(
                outcome_model,
                FixedSingleSampleModel,
            ):
                gen_metadata["outcome_model_fixed_draw_weights"] = outcome_model.w
        return gen_metadata

    @copy_doc(TorchModel.best_point)
    @single_surrogate_only
    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> Optional[Tensor]:
        try:
            return self.surrogate.best_in_sample_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )[0]
        except ValueError:
            return None

    @copy_doc(TorchModel.evaluate_acquisition_function)
    def evaluate_acquisition_function(
        self,
        X: Tensor,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        return acqf.evaluate(X=X)

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        # Will fail if metric_names exist across multiple models
        surrogate_labels = (
            [
                label
                for label, spec in self.surrogate_specs.items()
                if any(metric in spec.outcomes for metric in metric_names)
            ]
            if len(self.surrogates) > 1
            else [*self.surrogates.keys()]
        )
        if len(surrogate_labels) != 1:
            raise UserInputError(
                "May not cross validate multiple Surrogates at once. Please input "
                f"metric_names that exist on one Surrogate. {metric_names} spans "
                f"{surrogate_labels}"
            )
        surrogate_label = surrogate_labels[0]

        current_surrogates = self.surrogates
        # If we should be refitting but not warm-starting the refit, set
        # `state_dicts` to None to avoid loading it.
        state_dicts = (
            None
            if self.refit_on_cv and not self.warm_start_refit
            else {
                label: deepcopy(surrogate.model.state_dict())
                for label, surrogate in current_surrogates.items()
            }
        )

        # Temporarily set `_surrogates` to cloned surrogates to set
        # the training data on cloned surrogates to train set and
        # use it to predict the test point.
        surrogate_clones = {
            label: surrogate.clone_reset()
            for label, surrogate in self.surrogates.items()
        }
        self._surrogates = surrogate_clones
        # Remove the robust_digest since we do not want to use perturbations here.
        search_space_digest = dataclasses.replace(
            search_space_digest,
            robust_digest=None,
        )

        try:
            self.fit(
                datasets=datasets,
                metric_names=metric_names,
                search_space_digest=search_space_digest,
                state_dicts=state_dicts,
                refit=self.refit_on_cv,
                **kwargs,
            )
            X_test_prediction = self.predict_from_surrogate(
                surrogate_label=surrogate_label, X=X_test
            )
        finally:
            # Reset the surrogates back to this model's surrogate, make
            # sure the cloned surrogate doesn't stay around if fit or
            # predict fail.
            self._surrogates = current_surrogates
        return X_test_prediction

    @property
    def dtype(self) -> torch.dtype:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        dtypes = {
            label: surrogate.dtype for label, surrogate in self.surrogates.items()
        }

        dtypes_list = list(dtypes.values())
        if dtypes_list.count(dtypes_list[0]) != len(dtypes_list):
            raise NotImplementedError(
                f"Expected all Surrogates to have same dtype, found {dtypes}"
            )

        return dtypes_list[0]

    @property
    def device(self) -> torch.device:
        """Torch device type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """

        devices = {
            label: surrogate.device for label, surrogate in self.surrogates.items()
        }

        devices_list = list(devices.values())
        if devices_list.count(devices_list[0]) != len(devices_list):
            raise NotImplementedError(
                f"Expected all Surrogates to have same device, found {devices}"
            )

        return devices_list[0]

    def _instantiate_acquisition(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Acquisition:
        """Set a BoTorch acquisition function class for this model if needed and
        instantiate it.

        Returns:
            A BoTorch ``AcquisitionFunction`` instance.
        """
        if not self._botorch_acqf_class:
            if torch_opt_config.risk_measure is not None:
                # TODO[T131759261]: Implement selection of acqf for robust opt.
                # This will depend on the properties of the robust search space and
                # the risk measure being used.
                raise NotImplementedError
            self._botorch_acqf_class = choose_botorch_acqf_class(
                pending_observations=torch_opt_config.pending_observations,
                outcome_constraints=torch_opt_config.outcome_constraints,
                linear_constraints=torch_opt_config.linear_constraints,
                fixed_features=torch_opt_config.fixed_features,
                objective_thresholds=torch_opt_config.objective_thresholds,
                objective_weights=torch_opt_config.objective_weights,
            )

        return self.acquisition_class(
            surrogates=self.surrogates,
            botorch_acqf_class=self.botorch_acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acq_options,
        )

    def feature_importances(self) -> np.ndarray:
        """Compute feature importances from the model.

        Caveat: This assumes the following:
            1. There is a single surrogate model (potentially a `ModelList`).
            2. We can get model lengthscales from `covar_module.base_kernel.lengthscale`

        Returns:
            The feature importances as a numpy array of size len(metrics) x 1 x dim
            where each row sums to 1.
        """
        if list(self.surrogates.keys()) != [Keys.ONLY_SURROGATE]:
            raise NotImplementedError("Only support a single surrogate model for now")
        surrogate = self.surrogates[Keys.ONLY_SURROGATE]
        return get_feature_importances_from_botorch_model(model=surrogate.model)

    @property
    def search_space_digest(self) -> SearchSpaceDigest:
        if self._search_space_digest is None:
            raise RuntimeError(
                "`search_space_digest` is not initialized. Must `fit` the model first."
            )
        return self._search_space_digest

    @search_space_digest.setter
    def search_space_digest(self, value: SearchSpaceDigest) -> None:
        raise RuntimeError("Setting search_space_digest manually is disallowed.")

    @property
    def outcomes_by_surrogate_label(self) -> Dict[str, List[str]]:
        """Retuns a dictionary mapping from surrogate label to a list of outcomes."""
        outcomes_by_surrogate_label = {}
        for k, v in self.surrogates.items():
            outcomes_by_surrogate_label[k] = v.outcomes
        return outcomes_by_surrogate_label
