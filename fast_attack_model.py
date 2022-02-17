import torch.fx as fx
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Optional, Type, Dict, Any, Tuple, Iterable
import torch
from torch import nn
import copy


class FastAttackModel(nn.Module):
    """
    A wrapper around a model which can be used in adversarial attacks to run faster.
    It uses a few tricks to get significant (3-4x) speedup:
     * Automatic mixed-precision
     * No calculation of parameter gradients
     * Fusion of conv and batchnorm layers into a single conv layer
     * Using NWHC layout
    """

    fast_model: Optional[nn.Module]

    def __init__(self, model, fuse_bn=True, amp=True):
        super().__init__()
        self.model = model
        self.fast_model = None

        self.fuse_bn = fuse_bn
        self.amp = amp

    def update(self):
        """
        This should be called to update this model's state from the wrapped model.
        Always call this before running an adversarial attack.
        """

        if self.fuse_bn:
            self.fast_model = fuse(self.model, inplace=False)
        else:
            self.fast_model = copy.deepcopy(self.model)

        if self.amp:
            self.fast_model = self.fast_model.to(memory_format=torch.channels_last)

        for param in self.fast_model.parameters():
            param.requires_grad = False

        self.fast_model.training = self.model.training
        self.training = self.model.training

    def forward(self, *args, **kwargs):
        if self.fast_model is None:
            raise RuntimeError(
                "Need to call update() on FastAttackerModel before using it."
            )

        with torch.cuda.amp.autocast(self.amp):
            out = self.fast_model(*args, **kwargs)
        return out.float()


# Code from here down is taken from
# https://github.com/pytorch/pytorch/blob/orig/release/1.8/torch/fx/experimental/fuser.py


def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


# Works for length 2 patterns with 2 modules
def matches_module_pattern(
    pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]
):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != "call_module":
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(
    node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: torch.nn.Module, inplace=True) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)
