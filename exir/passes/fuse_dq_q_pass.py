# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


class FuseDQandQPass(ExportPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            ):
                if all(
                    user.target
                    == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
                    for user in list(node.users)
                ):
                    for user in list(node.users):
                        # Drop the input arg and check that the qparams are the same.
                        qparams_dq = list(node.args)[1:]
                        qparams_q = list(user.args)[1:]
                        if qparams_dq != qparams_q:
                            continue
                        user.replace_all_uses_with(node.args[0])

        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()
        return PassResult(graph_module, True)
