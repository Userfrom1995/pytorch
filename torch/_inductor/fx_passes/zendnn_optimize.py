# mypy: allow-untyped-defs
from torch._inductor.pattern_matcher import stable_topological_sort

from .zendnn_op_replacements import replace_with_zendnn_ops


def optimize(graph):
    # replace aten ops with zendnn ops
    opt_graph = replace_with_zendnn_ops(graph)
    # topological-sort, lint and recompile
    stable_topological_sort(opt_graph.graph)
    opt_graph.graph.lint()
    opt_graph.recompile()
    return opt_graph
