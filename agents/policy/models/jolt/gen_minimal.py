#!/usr/bin/env python3
"""
Generate a minimal ONNX model for authorization that fits within prover's MAX_TENSOR_SIZE=1024.
Architecture: 64 -> 16 -> 4
Max weight matrix: 64*16 = 1024 (exactly at limit)
"""

import numpy as np
import json

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available, trying torch export...")

# Dimensions
INPUT_SIZE = 64
HIDDEN_SIZE = 16
NUM_CLASSES = 4

if ONNX_AVAILABLE:
    # Create model directly with ONNX builder
    np.random.seed(42)

    # Initialize weights with Xavier initialization
    fc1_weight = (np.random.randn(HIDDEN_SIZE, INPUT_SIZE) * np.sqrt(2.0 / (INPUT_SIZE + HIDDEN_SIZE))).astype(np.float32)
    fc1_bias = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    fc2_weight = (np.random.randn(NUM_CLASSES, HIDDEN_SIZE) * np.sqrt(2.0 / (HIDDEN_SIZE + NUM_CLASSES))).astype(np.float32)
    fc2_bias = np.zeros(NUM_CLASSES, dtype=np.float32)

    # Create initializers (weights)
    fc1_weight_init = numpy_helper.from_array(fc1_weight, name='fc1.weight')
    fc1_bias_init = numpy_helper.from_array(fc1_bias, name='fc1.bias')
    fc2_weight_init = numpy_helper.from_array(fc2_weight, name='fc2.weight')
    fc2_bias_init = numpy_helper.from_array(fc2_bias, name='fc2.bias')

    # Create nodes
    # MatMul: input @ fc1_weight.T
    matmul1 = helper.make_node('MatMul', ['input', 'fc1_weight_t'], ['matmul1_out'])
    add1 = helper.make_node('Add', ['matmul1_out', 'fc1.bias'], ['add1_out'])
    relu = helper.make_node('Relu', ['add1_out'], ['relu_out'])
    matmul2 = helper.make_node('MatMul', ['relu_out', 'fc2_weight_t'], ['matmul2_out'])
    add2 = helper.make_node('Add', ['matmul2_out', 'fc2.bias'], ['output'])

    # Create transposed weight tensors for MatMul
    fc1_weight_t_init = numpy_helper.from_array(fc1_weight.T, name='fc1_weight_t')
    fc2_weight_t_init = numpy_helper.from_array(fc2_weight.T, name='fc2_weight_t')

    # Create graph
    graph = helper.make_graph(
        [matmul1, add1, relu, matmul2, add2],
        'auth_model',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, INPUT_SIZE])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, NUM_CLASSES])],
        [fc1_weight_t_init, fc1_bias_init, fc2_weight_t_init, fc2_bias_init]
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 6

    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, 'network.onnx')
else:
    # Fallback to torch
    import torch
    import torch.nn as nn

    class MinimalAuthModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = MinimalAuthModel()
    model.eval()

    dummy_input = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(
        model, dummy_input, 'network.onnx',
        export_params=True, opset_version=11,
        input_names=['input'], output_names=['output']
    )

print(f"✅ Created minimal ONNX model: network.onnx")
print(f"   Architecture: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {NUM_CLASSES}")
print(f"   Max weight matrix: {INPUT_SIZE * HIDDEN_SIZE} elements (within 1024 limit)")

# Also update vocab.json with feature mappings
vocab = {
    "vocab_mapping": {},
    "feature_mapping": {}
}

# Feature buckets that sum to 64
features = [
    ("budget", 16),   # indices 0-15
    ("trust", 8),     # indices 16-23
    ("amount", 16),   # indices 24-39
    ("category", 4),  # indices 40-43
    ("velocity", 8),  # indices 44-51
    ("day", 8),       # indices 52-59
    ("time", 4)       # indices 60-63
]

idx = 0
for name, count in features:
    vocab["feature_mapping"][name] = [f"{name}_{i}" for i in range(count)]
    for i in range(count):
        key = f"{name}_{i}"
        vocab["vocab_mapping"][key] = {"index": idx, "feature_type": name}
        idx += 1

with open('vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

print(f"✅ Updated vocab.json with {idx} feature mappings")

# Create meta.json
meta = {
    "model_type": "authorization_classifier",
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": NUM_CLASSES,
    "architecture": "64 -> 16 -> 4 MLP",
    "max_tensor_size": INPUT_SIZE * HIDDEN_SIZE,
    "prover_compatible": True
}

with open('meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f"✅ Created meta.json")
