#!/usr/bin/env python

from __future__ import annotations

import argparse
from timeit import default_timer as timer

import torch
from torch.distributions import StickBreakingTransform
from torchtree import Parameter, TransformedParameter
from torchtree.evolution.alignment import Alignment, Sequence
from torchtree.evolution.datatype import NucleotideDataType
from torchtree.evolution.io import read_tree, read_tree_and_alignment
from torchtree.evolution.taxa import Taxa, Taxon
from torchtree.evolution.tree_model import heights_from_branch_lengths

from torchtree_physher import (
    ConstantCoalescentModel,
    ConstantSiteModel,
    PiecewiseLinearCoalescentGridModel,
    ReparameterizedTimeTreeModel,
    TimeTreeModel,
    UnRootedTreeModel,
)
from torchtree_physher.substitution_model import GTR, JC69
from torchtree_physher.tree_likelihood import (
    TreeLikelihoodFunction,
    TreeLikelihoodModel,
)
from torchtree_physher.tree_model import GeneralNodeHeightTransform


def benchmark(f):
    def timed(replicates, *args):
        start = timer()
        for _ in range(replicates):
            out = f(*args)
        end = timer()
        total_time = end - start
        return total_time, out

    return timed


@benchmark
def tree_likelihood_fn(inst, models):
    tree_model = models[0]
    blens = tree_model.branch_lengths()
    fn = TreeLikelihoodFunction.apply
    log_prob = fn(inst, models, blens, None, None, None, None, None)
    return log_prob


@benchmark
def gradient_tree_likelihood(inst, models):
    tree_model = models[0]
    blens = tree_model.branch_lengths()
    fn = TreeLikelihoodFunction.apply
    log_prob = fn(inst, models, blens, None, None, None, None, None)
    log_prob.backward()
    return log_prob


@benchmark
def tree_likelihood_gtr_fn(inst, models):
    tree_model = models[0]
    subst_model = models[1]
    blens = tree_model.branch_lengths()
    for p in subst_model.parameters():
        p.fire_parameter_changed()
    rates = subst_model.rates.unsqueeze(0)
    frequencies = subst_model.frequencies.unsqueeze(0)
    fn = TreeLikelihoodFunction.apply
    log_prob = fn(inst, models, blens, None, rates, frequencies, None, None)
    return log_prob


@benchmark
def gradient_tree_likelihood_gtr(inst, models):
    tree_model = models[0]
    subst_model = models[1]
    blens = tree_model.branch_lengths()
    for p in subst_model.parameters():
        p.fire_parameter_changed()
    rates = subst_model.rates.unsqueeze(0)
    frequencies = subst_model.frequencies.unsqueeze(0)

    fn = TreeLikelihoodFunction.apply
    log_prob = fn(inst, models, blens, None, rates, frequencies, None, None)
    log_prob.backward()
    return log_prob


def unrooted_treelikelihood(args, subst_model):
    tree, dna = read_tree_and_alignment(args.tree, args.input, True, True)
    sequences = []
    taxon_list = []
    for taxon, seq in dna.items():
        sequences.append(Sequence(taxon.label, str(seq)))
        taxon_list.append(Taxon(taxon.label, None))
    taxa = Taxa(None, taxon_list)
    branch_lengths = torch.tensor(
        [
            float(node.edge_length) * args.scaler
            for node in sorted(
                list(tree.postorder_node_iter())[:-1], key=lambda x: x.index
            )
        ],
    )

    child_1, child_2 = tree.seed_node.child_node_iter()
    branch_lengths[child_1.index] += branch_lengths[child_2.index]
    branch_lengths = Parameter(
        "blens",
        branch_lengths[:-1].unsqueeze(0),
    )
    branch_lengths.tensor = torch.clamp(branch_lengths.tensor, min=1.0e-6)

    site_model = ConstantSiteModel(None)
    tree_model = UnRootedTreeModel(None, tree, taxa, branch_lengths)
    alignment = Alignment(None, sequences, taxa, NucleotideDataType("nuc"))
    tree_likelihood = TreeLikelihoodModel(
        None,
        alignment,
        tree_model,
        subst_model,
        site_model,
        use_tip_states=False,
        use_ambiguities=True,
    )

    print("treelikelihood")

    total_time, log_prob = tree_likelihood_fn(
        args.replicates,
        tree_likelihood.inst,
        [
            tree_likelihood.tree_model,
            tree_likelihood.subst_model,
            tree_likelihood.site_model,
        ],
    )
    print(f"  {args.replicates} evaluations: {total_time} ({log_prob})")

    branch_lengths.requires_grad = True
    for p in subst_model.parameters():
        p.requires_grad = True

    if isinstance(subst_model, JC69):
        grad_total_time, grad_log_prob = gradient_tree_likelihood(
            args.replicates,
            tree_likelihood.inst,
            [
                tree_likelihood.tree_model,
                tree_likelihood.subst_model,
                tree_likelihood.site_model,
            ],
        )
    else:
        grad_total_time, grad_log_prob = gradient_tree_likelihood_gtr(
            args.replicates,
            tree_likelihood.inst,
            [
                tree_likelihood.tree_model,
                tree_likelihood.subst_model,
                tree_likelihood.site_model,
            ],
        )
    print(
        f"  {args.replicates} gradient evaluations: {grad_total_time} ({grad_log_prob}"
    )

    if args.output:
        name = type(subst_model).__name__
        args.output.write(
            f"treelikelihood{name},evaluation,off,{total_time},"
            f"{log_prob.squeeze().item()}\n"
        )
        args.output.write(
            f"treelikelihood{name},gradient,off,{grad_total_time},"
            f"{grad_log_prob.squeeze().item()}\n"
        )


def ratio_transform_jacobian(args):
    tree = read_tree(args.tree, True, True)
    taxa = []
    for node in tree.leaf_node_iter():
        taxa.append(Taxon(node.taxon.label, {"date": node.date}))
    taxa_count = len(taxa)
    ratios_root_height = Parameter(
        "internal_heights", torch.tensor([0.5] * (taxa_count - 1) + [20])
    )
    tree_model = ReparameterizedTimeTreeModel(
        "tree", tree, Taxa("taxa", taxa), ratios_root_height
    )

    ratios_root_height.tensor = tree_model.transform.inv(
        heights_from_branch_lengths(tree)
    )
    internal_heights = tree_model.transform(ratios_root_height.tensor)
    tree_model._internal_heights.tensor = ratios_root_height.tensor.unsqueeze(0)
    tree_model.update(0)
    tree_model()

    @benchmark
    def fn():
        return torch.tensor([tree_model.inst.transform_jacobian()])

    @benchmark
    def fn_grad():
        return torch.tensor(tree_model.inst.gradient_transform_jacobian())

    total_time, log_det_jac = fn(args.replicates)
    print(f"  {args.replicates} evaluations: {total_time} ({log_det_jac})")

    internal_heights.requires_grad = True
    grad_total_time, grad_log_det_jac = fn_grad(args.replicates)
    print(f"  {args.replicates} gradient evaluations: {grad_total_time}")

    if args.output:
        args.output.write(
            f"ratio_transform_jacobian,evaluation,off,{total_time},"
            f"{log_det_jac.squeeze().item()}\n"
        )
        args.output.write(f"ratio_transform_jacobian,gradient,off,{grad_total_time},\n")


def ratio_transform(args):
    replicates = args.replicates
    tree = read_tree(args.tree, True, True)
    taxa_count = len(tree.taxon_namespace)
    taxa = []
    for node in tree.leaf_node_iter():
        taxa.append(Taxon(str(node.taxon).replace("'", ""), {"date": node.date}))
    ratios_root_height = Parameter(
        "internal_heights", torch.tensor([0.5] * (taxa_count - 2) + [10])
    )
    tree_model = ReparameterizedTimeTreeModel(
        "tree", tree, Taxa("taxa", taxa), ratios_root_height
    )

    ratios_root_height.tensor = tree_model.transform.inv(
        heights_from_branch_lengths(tree)
    )

    @benchmark
    def fn():
        return GeneralNodeHeightTransform(tree_model.inst)(
            tree_model._internal_heights.tensor
        )

    @benchmark
    def fn_grad():
        heights = GeneralNodeHeightTransform(tree_model.inst)(
            tree_model._internal_heights.tensor
        )
        heights.backward(torch.ones_like(tree_model._internal_heights))
        tree_model._internal_heights.grad.data.zero_()
        return heights

    total_time, heights = fn(args.replicates)
    print(f"  {replicates} evaluations: {total_time}")

    tree_model._internal_heights.requires_grad = True
    grad_total_time, heights = fn_grad(args.replicates)
    print(f"  {replicates} gradient evaluations: {grad_total_time}")

    if args.output:
        args.output.write(f"ratio_transform,evaluation,off,{total_time},\n")
        args.output.write(f"ratio_transform,gradient,off,{grad_total_time},\n")


def create_time_tree_model(args):
    tree = read_tree(args.tree, True, True)
    taxa = []
    for node in tree.leaf_node_iter():
        taxa.append(Taxon(str(node.taxon).replace("'", ""), {"date": node.date}))
    internal_heights = Parameter("internal_heights", heights_from_branch_lengths(tree))
    tree_model = TimeTreeModel("tree", tree, Taxa("taxa", taxa), internal_heights)
    return tree_model


def constant_coalescent(args):
    tree_model = create_time_tree_model(args)
    internal_heights = tree_model._internal_heights
    pop_size = Parameter("constant", torch.tensor([4.0]))
    coalescent_model = ConstantCoalescentModel(None, pop_size, tree_model)

    @benchmark
    def fn():
        coalescent_model.lp_needs_update = True
        return coalescent_model()

    @benchmark
    def fn_grad():
        coalescent_model.lp_needs_update = True
        log_p = coalescent_model()
        log_p.backward()
        internal_heights.tensor.grad.data.zero_()
        pop_size.grad.data.zero_()
        return log_p

    # total_time, log_p = fn(args.replicates)
    total_time, log_p = fn(args.replicates)
    print(f"  {args.replicates} evaluations: {total_time} {log_p}")

    internal_heights.requires_grad = True
    pop_size.requires_grad = True
    grad_total_time, grad_log_p = fn_grad(args.replicates)
    print(f"  {args.replicates} gradient evaluations: {grad_total_time}")

    if args.output:
        args.output.write(
            f"coalescent,evaluation,off,{total_time},{log_p.squeeze().item()}\n"
        )
        args.output.write(
            f"coalescent,gradient,off,{grad_total_time},{grad_log_p.squeeze().item()}\n"
        )

    if args.debug:
        coalescent_model.lp_needs_update = True
        log_p = coalescent_model()
        log_p.backward()
        print("gradient internal heights: ", internal_heights.grad)
        print("gradient pop size: ", pop_size.grad)


def skyglide_coalescent(args):
    tree_model = create_time_tree_model(args)
    internal_heights = tree_model._internal_heights
    pop_size = Parameter(None, torch.arange(0.0, args.intervals, step=1.0) + 5)
    if args.cutoff is None:
        cutoff = 1.1 * internal_heights.tensor[-1].item()
    else:
        cutoff = args.cutoff
    grid = Parameter(None, torch.linspace(0, cutoff, pop_size.shape[-1])[1:])
    coalescent_model = PiecewiseLinearCoalescentGridModel(
        "skyglide", pop_size, grid, tree_model
    )

    @benchmark
    def fn():
        coalescent_model.lp_needs_update = True
        return coalescent_model()

    @benchmark
    def fn_grad():
        coalescent_model.lp_needs_update = True
        log_p = coalescent_model()
        log_p.backward()
        internal_heights.tensor.grad.data.zero_()
        pop_size.grad.data.zero_()
        return log_p

    total_time, log_p = fn(args.replicates)
    print(f"  {args.replicates} evaluations: {total_time} {log_p}")

    internal_heights.requires_grad = True
    pop_size.requires_grad = True
    grad_total_time, grad_log_p = fn_grad(args.replicates)
    print(f"  {args.replicates} gradient evaluations: {grad_total_time}")

    if args.output:
        args.output.write(
            f"skyglide,evaluation,off,{total_time},{log_p.squeeze().item()}\n"
        )
        args.output.write(f"skyglide,gradient,off,{grad_total_time},\n")

    if args.debug:
        coalescent_model.lp_needs_update = True
        log_p = coalescent_model()
        log_p.backward()
        print("gradient internal heights: ", internal_heights.grad)
        print("gradient pop size: ", pop_size.grad)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="""Alignment file""")
parser.add_argument("-t", "--tree", required=True, help="""Tree file""")
parser.add_argument(
    "-r",
    "--replicates",
    required=True,
    type=int,
    help="""Number of replicates""",
)
parser.add_argument(
    "-o",
    "--output",
    type=argparse.FileType("w"),
    default=None,
    help="""csv output file""",
)
parser.add_argument(
    "-s",
    "--scaler",
    type=float,
    default=1.0,
    help="""scale branch lengths""",
)
parser.add_argument("--debug", action="store_true", help="""Debug mode""")
parser.add_argument(
    "--gtr",
    action="store_true",
    help="""Include gradient calculation of GTR parameters""",
)
parser.add_argument(
    "--cutoff",
    type=float,
    help="""cutoff for piecewise coalescent""",
)
parser.add_argument(
    "--intervals",
    type=int,
    help="""number of intervals in piecewise coalescent""",
)
args = parser.parse_args()


if args.output:
    args.output.write("function,mode,JIT,time,logprob\n")

print("Tree likelihood unrooted:")
unrooted_treelikelihood(args, JC69("jc"))
print()

if args.gtr:
    print("Tree likelihood unrooted:")
    unrooted_treelikelihood(
        args,
        GTR(
            "gtr",
            TransformedParameter(
                "rates",
                Parameter("rates.unres", torch.full((5,), 0.0)),
                StickBreakingTransform(),
            ),
            TransformedParameter(
                "frequencies",
                Parameter("frequencies.unres", torch.full((3,), 0.0)),
                StickBreakingTransform(),
            ),
        ),
    )
    print()

print("Height transform log det Jacobian:")
ratio_transform_jacobian(args)
print()

print("Node height transform:")
ratio_transform(args)

print()
print("Constant coalescent:")
constant_coalescent(args)

if args.intervals is not None:
    print()
    print("Piecewise linear coalescent:")
    skyglide_coalescent(args)

if args.output:
    args.output.close()
