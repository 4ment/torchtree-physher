import torch

from physherpy.physher import TreeLikelihoodModel as PhysherTreeLikelihood
from torchtree import TransformedParameter
from torchtree.core.model import CallableModel
from torchtree.core.utils import JSONParseError, process_object
from torchtree.evolution.alignment import Alignment, read_fasta_sequences
from torchtree.evolution.branch_model import BranchModel
from torchtree.evolution.site_model import SiteModel
from torchtree.evolution.site_pattern import SitePattern
from torchtree.evolution.substitution_model.abstract import SubstitutionModel
from torchtree.evolution.tree_model import TreeModel
from torchtree.typing import ID


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
    ):
        super().__init__(id_)
        self.tree_model = tree_model
        self.subst_model = subst_model
        self.site_model = site_model
        self.clock_model = clock_model
        clock_inst = clock_model.inst if clock_model is not None else None

        self.inst = PhysherTreeLikelihood(
            alignment,
            tree_model.inst,
            subst_model.inst,
            site_model.inst,
            clock_inst,
            use_ambiguities,
        )

    def _call(self, *args, **kwargs) -> torch.Tensor:
        fn = TreeLikelihoodFunction.apply
        models = [self.tree_model, self.subst_model, self.site_model]
        if self.clock_model is not None:
            models.append(self.clock_model)

        clock_rate = None
        mu = None
        weibull_shape = (
            None if self.site_model.rates().shape[-1] == 1 else self.site_model.shape
        )

        if self.clock_model:
            branch_parameters = self.tree_model._internal_heights.tensor
            clock_rate = self.clock_model._rates.tensor
        else:
            branch_parameters = self.tree_model.branch_lengths()

        subst_rates = None
        subst_frequencies = None
        if isinstance(self.subst_model._frequencies, TransformedParameter):
            subst_frequencies = self.subst_model._frequencies.x.tensor

        if hasattr(self.subst_model, '_rates') and isinstance(
            self.subst_model._rates, TransformedParameter
        ):
            subst_rates = self.subst_model._rates.x.tensor
        elif hasattr(self.subst_model, '_kappa'):
            subst_rates = self.subst_model.kappa

        return fn(
            self.inst,
            models,
            branch_parameters,
            clock_rate,
            subst_rates,
            subst_frequencies,
            weibull_shape,
            mu,
        )

    def handle_model_changed(self, model, obj, index):
        self.fire_model_changed()

    def handle_parameter_changed(self, variable, index, event):
        pass

    @property
    def sample_shape(self) -> torch.Size:
        return max([model.sample_shape for model in self._models.values()], key=len)

    @classmethod
    def from_json(cls, data, dic):
        id_ = data['id']
        tree_model = process_object(data[TreeModel.tag], dic)
        site_model = process_object(data[SiteModel.tag], dic)
        subst_model = process_object(data[SubstitutionModel.tag], dic)

        # Ignore site_pattern and parse alignment instead

        # alignment is a reference to an object already parsed
        if isinstance(data[SitePattern.tag]['alignment'], str):
            alignment = dic[data[SitePattern.tag]['alignment']]
        # alignment contains a file entry
        elif 'file' in data[SitePattern.tag]['alignment']:
            alignment = read_fasta_sequences(data[SitePattern.tag]['alignment']['file'])
        # alignment contains a dictionary of sequences
        elif 'sequences' in data[SitePattern.tag]['alignment']:
            alignment = Alignment.from_json(data[SitePattern.tag]['alignment'], dic)
        else:
            raise JSONParseError('site_pattern is misspecified')

        use_ambiguities = data.get('use_ambiguities', False)
        clock_model = None
        if BranchModel.tag in data:
            clock_model = process_object(data[BranchModel.tag], dic)
        return cls(
            id_,
            alignment,
            tree_model,
            subst_model,
            site_model,
            clock_model,
            use_ambiguities,
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
        weibull_shape=None,
        mu=None,
    ):
        ctx.inst = inst
        ctx.save_for_backward(
            branch_lengths,
            clock_rates,
            subst_rates,
            subst_frequencies,
            weibull_shape,
            mu,
        )

        log_probs = []
        grads = []
        for i in range(branch_lengths.shape[0]):
            for model in models:
                model.update(i)
            log_probs.append(torch.tensor([inst.log_likelihood()]))
            if branch_lengths.requires_grad:
                grads.append(torch.tensor(inst.gradient()))
        ctx.grads = torch.stack(grads) if branch_lengths.requires_grad else None
        return torch.stack(log_probs)

    @staticmethod
    def backward(ctx, grad_output):
        (
            branch_lengths,
            clock_rates,
            subst_rates,
            subst_frequencies,
            weibull_shape,
            mu,
        ) = ctx.saved_tensors

        branch_grad = ctx.grads[
            ..., : branch_lengths.shape[-1]
        ] * grad_output.unsqueeze(-1)
        offset = branch_lengths.shape[-1]

        if clock_rates is not None:
            clock_rate_grad = ctx.grads[
                ..., offset : (offset + clock_rates.shape[-1])
            ] * grad_output.unsqueeze(-1)
            offset += clock_rates.shape[-1]
        else:
            clock_rate_grad = None

        if subst_rates is not None:
            subst_rates_grad = ctx.grads[
                ..., offset : (offset + subst_rates.shape[-1])
            ] * grad_output.unsqueeze(-1)
            offset += subst_rates.shape[-1]
        else:
            subst_rates_grad = None

        if subst_frequencies is not None:
            subst_frequencies_grad = ctx.grads[
                ..., offset : (offset + subst_frequencies.shape[-1])
            ] * grad_output.unsqueeze(-1)
            offset += subst_frequencies.shape[-1]
        else:
            subst_frequencies_grad = None

        if weibull_shape is not None:
            weibull_grad = ctx.grads[
                ..., offset : (offset + weibull_shape.shape[-1])
            ] * grad_output.unsqueeze(-1)
            offset += weibull_shape.shape[-1]
        else:
            weibull_grad = None

        if mu is not None:
            mu_grad = ctx.grads[
                ..., offset : (offset + mu.shape[-1])
            ] * grad_output.unsqueeze(-1)
            offset += mu.shape[-1]
        else:
            mu_grad = None

        return (
            None,  # inst
            None,  # models
            branch_grad,
            clock_rate_grad,
            subst_rates_grad,
            subst_frequencies_grad,
            weibull_grad,
            mu_grad,
        )
