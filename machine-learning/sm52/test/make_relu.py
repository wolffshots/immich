import sys

import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "relu", [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9
onnx.save(model, sys.argv[1])
print("wrote", sys.argv[1])
