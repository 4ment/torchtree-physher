import torch
from torchtree import TransformedParameter
from torchtree.core.model import CallableModel
from torchtree.core.utils import JSONParseError, process_object, string_to_list_index
from torchtree.evolution.alignment import Alignment, Sequence, read_fasta_sequences
from torchtree.evolution.branch_model import BranchModel
from torchtree.evolution.site_model import SiteModel, UnivariateDiscretizedSiteModel
from torchtree.evolution.site_pattern import SitePattern
from torchtree.evolution.substitution_model.abstract import SubstitutionModel
from torchtree.evolution.tree_model import TreeModel
from torchtree.typing import ID

import torchtree_physher.physher.tree_likelihood_gradient_flags as flags
from torchtree_physher.physher import TreeLikelihoodModel as PhysherTreeLikelihood
from torchtree_physher.utils import flatten_2D


class TreeLikelihoodModel(CallableModel):
    def __init__(
        self,
        id_: ID,
        alignment: Alignment,
        tree_model: TreeModel,
        subst_model: SubstitutionModel,
        site_model: SiteModel,
        clock_model: BranchModel = None,
        use_ambiguities=False,
        use_tip_states=False,
        use_sse=True,
        include_jacobian=False,
    ):
        super().__init__(id_)
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        clock_inst = clock_model.inst if clock_model is not None else None

        if isinstance(alignment, tuple):
            self.inst = PhysherTreeLikelihood(
                *alignment,
                tree_model.inst,
                subst_model.inst,
                site_model.inst,
                clock_inst,
                use_ambiguities,
                use_tip_states,
                include_jacobian,
            )
        else:
            self.inst = PhysherTreeLikelihood(
                alignment,
                tree_model.inst,
                subst_model.inst,
                site_model.inst,
                clock_inst,
                use_ambiguities,
                use_tip_states,
                include_jacobian,
            )
        if not use_sse:
            self.inst.enable_sse(use_sse)

    def _call(self, *args, **kwargs) -> torch.Tensor:
        fn = TreeLikelihoodFunction.apply
        models = [self.tree_model, self.subst_model, self.site_model]
        if self.clock_model is not None:
            models.append(self.clock_model)

        clock_rate = None
        mu = None
        site_parameter = (
            self.site_model._parameter.tensor
            if isinstance(self.site_model, UnivariateDiscretizedSiteModel)
            else None
        )

        pinv = (
            self.site_model._invariant.tensor
            if hasattr(self.site_model, "_invariant")
            and self.site_model._invariant is not None
            else None
        )

        if self.site_model._mu is not None:
            mu = self.site_model._mu.tensor

        if self.clock_model:
            branch_parameters = self.tree_model._internal_heights.tensor
            clock_rate = self.clock_model._rates.tensor
        else:
            branch_parameters = self.tree_model.branch_lengths()

        subst_rates = None
        subst_frequencies = None
        if isinstance(self.subst_model._frequencies, TransformedParameter):
            subst_frequencies = self.subst_model._frequencies.tensor

        if hasattr(self.subst_model, "_rates") and isinstance(
            self.subst_model._rates, TransformedParameter
        ):
            subst_rates = self.subst_model._rates.tensor
        elif hasattr(self.subst_model, "_kappa") and isinstance(
            self.subst_model._kappa, TransformedParameter
        ):
            subst_rates = self.subst_model.kappa

        return fn(
            self.inst,
            models,
            branch_parameters,
            clock_rate,
            subst_rates,
            subst_frequencies,
            site_parameter,
            pinv,
            mu,
        )

    def handle_parameter_changed(self, variable, index, event):
        pass

    def _sample_shape(self) -> torch.Size:
        return max([model.sample_shape for model in self._models.values()], key=len)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data["id"]
        tree_model = process_object(data[TreeModel.tag], dic)
        site_model = process_object(data[SiteModel.tag], dic)
        subst_model = process_object(data[SubstitutionModel.tag], dic)

        use_ambiguities = data.get("use_ambiguities", False)
        use_tip_states = data.get("use_tip_states", False)
        use_sse = data.get("use_sse", True)
        include_jacobian = data.get("include_jacobian", False)

        clock_model = None
        if BranchModel.tag in data:
            clock_model = process_object(data[BranchModel.tag], dic)
            tree_model.zero_jacobian = include_jacobian

        # Ignore site_pattern and parse alignment instead

        if data[SitePattern.tag]["type"] == "torchtree_physher.AttributePattern":
            taxa = process_object(data[SitePattern.tag]["taxa"], dic)
            attribute_name = data[SitePattern.tag]["attribute"]
            taxon_list = [taxon.id for taxon in taxa]
            attribute_list = [taxon[attribute_name] for taxon in taxa]

            return cls(
                id_,
                (taxon_list, attribute_list),
                tree_model,
                subst_model,
                site_model,
                clock_model,
                use_ambiguities,
                use_tip_states,
                include_jacobian,
            )

        # alignment is a reference to an object already parsed
        elif isinstance(data[SitePattern.tag]["alignment"], str):
            alignment = dic[data[SitePattern.tag]["alignment"]]
        # alignment contains a file entry
        elif "file" in data[SitePattern.tag]["alignment"]:
            alignment = read_fasta_sequences(data[SitePattern.tag]["alignment"]["file"])
        # alignment contains a dictionary of sequences
        elif "sequences" in data[SitePattern.tag]["alignment"]:
            alignment = Alignment.from_json(data[SitePattern.tag]["alignment"], dic)
        else:
            raise JSONParseError("site_pattern is misspecified")

        if "indices" in data[SitePattern.tag]:
            indices = data[SitePattern.tag]["indices"]
            if indices is not None:
                list_of_indices = [
                    string_to_list_index(index_str) for index_str in indices.split(",")
                ]
                sequences_new = [""] * len(alignment)
                for index in list_of_indices:
                    for idx, seq in enumerate(alignment):
                        sequences_new[idx] += seq.sequence[index]
                alignment = [
                    Sequence(alignment[idx].taxon, seq)
                    for idx, seq in enumerate(sequences_new)
                ]

        return cls(
            id_,
            alignment,
            tree_model,
            subst_model,
            site_model,
            clock_model,
            use_ambiguities,
            use_tip_states,
            use_sse,
            include_jacobian,
        )


class TreeLikelihoodFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inst,
        models,
        branch_lengths,
        clock_rates=None,
        subst_rates=None,
        subst_frequencies=None,
        site_parameter=None,
        pinv=None,
        mu=None,
    ) -> torch.Tensor:
        """Evaluate log tree likelihood using physher

        :param ctx: context
        :param PhysherTreeLikelihood inst: physher tree likelihood instance
        :param models: list of models to update
        :type models: list
        :param torch.Tensor branch_lengths: branch length or ratios tensor
        :param torch.Tensor clock_rates: substitution rate tensor
        :param torch.Tensor subst_rates: transition rate matrix biases tensor
        :param torch.Tensor subst_frequencies: frequencies of the transition rate matrix
        :param torch.Tensor site_parameter: shape tensor of the Weibull distribution
        :param torch.Tensor pinv: proportion of invariant site tensor
        :param torch.Tensor mu: mu tensor
        :return:
        """
        ctx.inst = inst
        ctx.save_for_backward(
            branch_lengths,
            clock_rates,
            subst_rates,
            subst_frequencies,
            site_parameter,
            pinv,
            mu,
        )

        rate_need_update = True
        physher_flags = []
        if branch_lengths.requires_grad:
            physher_flags.append(flags.TREE_HEIGHT)
        if clock_rates is not None and clock_rates.requires_grad:
            physher_flags.append(flags.BRANCH_MODEL)
        if subst_rates is not None and subst_rates.requires_grad:
            physher_flags.append(flags.SUBSTITUTION_MODEL_RATES)
        if subst_frequencies is not None and subst_frequencies.requires_grad:
            physher_flags.append(flags.SUBSTITUTION_MODEL_FREQUENCIES)
        if (
            (site_parameter is not None and site_parameter.requires_grad)
            or (pinv is not None and pinv.requires_grad)
            or (mu is not None and mu.requires_grad)
        ):
            physher_flags.append(flags.SITE_MODEL)
        if len(physher_flags) > 0:
            inst.request_gradient(physher_flags)

        # Fixed clock rate
        if clock_rates is not None and clock_rates.dim() == 1:
            models[3].update(0)
            rate_need_update = False
        # Unrooted tree
        elif clock_rates is None:
            rate_need_update = False

        options = {"dtype": branch_lengths.dtype, "device": branch_lengths.device}

        log_probs = []
        grads = []
        branch_lengths_flatten = flatten_2D(branch_lengths)
        for i in range(branch_lengths_flatten.shape[0]):
            for model in models[:3]:
                model.update(i)
            if rate_need_update:
                models[3].update(i)

            log_probs.append(inst.log_likelihood())
            if len(physher_flags) > 0:
                grads.append(torch.tensor(inst.gradient(), **options))

        ctx.grads = None
        if len(grads) > 0:
            ctx.grads = torch.stack(grads).view(branch_lengths.shape[:-1] + (-1,))

        return torch.tensor(log_probs, **options).view(branch_lengths.shape[:-1] + (1,))

    @staticmethod
    def backward(ctx, grad_output) -> torch.Tensor:
        """Compute gradient using physher

        Derivatives returned by physher are concatenated in this order:
         - branch length or ratios
         - site model: weibull shape, pinv, mu
         - clock model: substitution rate(s)
         - substitution model: rate(s), frequencies

        :param ctx: context
        :param torch.Tensor grad_output:
        :return torch.Tensor: gradient
        """
        (
            branch_lengths,
            clock_rates,
            subst_rates,
            subst_frequencies,
            site_parameter,
            pinv,
            mu,
        ) = ctx.saved_tensors

        def extract_grad(param, offset):
            if param is not None and param.requires_grad:
                param_grad = (
                    ctx.grads[..., offset : offset + param.shape[-1]] * grad_output
                )
                offset += param.shape[-1]
            else:
                param_grad = None
            return param_grad, offset

        offset = 0
        branch_grad, offset = extract_grad(branch_lengths, offset)
        site_parameter_grad, offset = extract_grad(site_parameter, offset)
        pinv_grad, offset = extract_grad(pinv, offset)
        mu_grad, offset = extract_grad(mu, offset)
        clock_rates_grad, offset = extract_grad(clock_rates, offset)
        subst_rates_grad, offset = extract_grad(subst_rates, offset)
        subst_frequencies_grad, offset = extract_grad(subst_frequencies, offset)

        return (
            None,  # inst
            None,  # models
            branch_grad,
            clock_rates_grad,
            subst_rates_grad,
            subst_frequencies_grad,
            site_parameter_grad,
            pinv_grad,
            mu_grad,
        )
