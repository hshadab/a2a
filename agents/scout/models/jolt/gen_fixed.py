#!/usr/bin/env python3
"""
Fixed URL Quality Scorer Model for Jolt Atlas zkML

Exports using MatMul + Add (not Gemm) with transposed weights
to match the format expected by Jolt Atlas prover.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

INPUT_SIZE = 32
HIDDEN_SIZE = 16
NUM_CLASSES = 4
QUALITY_TIERS = ["HIGH", "MEDIUM", "LOW", "NOISE"]


class URLQualityScorerMLP(nn.Module):
    """MLP: 32 -> 16 -> 4"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def export_jolt_compatible_onnx(model, output_path='network.onnx'):
    """
    Export model to ONNX using MatMul + Add format (not Gemm)
    with transposed weights to match Jolt Atlas expectations.
    """
    print(f"Exporting Jolt-compatible ONNX to {output_path}")

    model.eval()

    # Get weights and biases
    fc1_weight = model.fc1.weight.detach().numpy()  # [16, 32]
    fc1_bias = model.fc1.bias.detach().numpy()       # [16]
    fc2_weight = model.fc2.weight.detach().numpy()  # [4, 16]
    fc2_bias = model.fc2.bias.detach().numpy()       # [4]

    # Transpose weights for MatMul: input @ weight_t = output
    # [1, 32] @ [32, 16] = [1, 16]
    fc1_weight_t = fc1_weight.T  # [32, 16]
    fc2_weight_t = fc2_weight.T  # [16, 4]

    print(f"  fc1_weight_t: {fc1_weight_t.shape}")  # Should be [32, 16]
    print(f"  fc2_weight_t: {fc2_weight_t.shape}")  # Should be [16, 4]

    # Create ONNX graph manually (like authorization model)
    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, INPUT_SIZE])

    # Output
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, NUM_CLASSES])

    # Initializers (weights and biases)
    fc1_weight_init = helper.make_tensor('fc1_weight_t', TensorProto.FLOAT,
                                          fc1_weight_t.shape, fc1_weight_t.flatten().tolist())
    fc1_bias_init = helper.make_tensor('fc1.bias', TensorProto.FLOAT,
                                        fc1_bias.shape, fc1_bias.tolist())
    fc2_weight_init = helper.make_tensor('fc2_weight_t', TensorProto.FLOAT,
                                          fc2_weight_t.shape, fc2_weight_t.flatten().tolist())
    fc2_bias_init = helper.make_tensor('fc2.bias', TensorProto.FLOAT,
                                        fc2_bias.shape, fc2_bias.tolist())

    # Nodes: MatMul -> Add -> Relu -> MatMul -> Add
    matmul1 = helper.make_node('MatMul', ['input', 'fc1_weight_t'], ['matmul1_out'])
    add1 = helper.make_node('Add', ['matmul1_out', 'fc1.bias'], ['add1_out'])
    relu = helper.make_node('Relu', ['add1_out'], ['relu_out'])
    matmul2 = helper.make_node('MatMul', ['relu_out', 'fc2_weight_t'], ['matmul2_out'])
    add2 = helper.make_node('Add', ['matmul2_out', 'fc2.bias'], ['output'])

    # Create graph
    graph = helper.make_graph(
        [matmul1, add1, relu, matmul2, add2],
        'url_quality_scorer',
        [input_tensor],
        [output_tensor],
        [fc1_weight_init, fc1_bias_init, fc2_weight_init, fc2_bias_init]
    )

    # Create model
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])

    # Validate
    onnx.checker.check_model(onnx_model)

    # Save
    onnx.save(onnx_model, output_path)
    print(f"  Saved to {output_path}")

    # Verify structure
    loaded = onnx.load(output_path)
    print(f"  Nodes: {[n.op_type for n in loaded.graph.node]}")
    print(f"  Initializers: {[(i.name, list(i.dims)) for i in loaded.graph.initializer]}")

    return True


def test_onnx_model(model, onnx_path):
    """Verify ONNX model produces same output as PyTorch"""
    import onnxruntime as ort

    model.eval()
    test_input = torch.randn(1, INPUT_SIZE)

    # PyTorch output
    with torch.no_grad():
        pytorch_out = model(test_input).numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {'input': test_input.numpy()})[0]

    # Compare
    diff = np.abs(pytorch_out - onnx_out).max()
    print(f"  Max diff between PyTorch and ONNX: {diff:.6f}")

    if diff < 1e-5:
        print("  ✓ ONNX model matches PyTorch!")
        return True
    else:
        print("  ✗ ONNX model differs from PyTorch!")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("URL Quality Scorer - Jolt Atlas Compatible Export")
    print("=" * 60)

    # Load trained model
    model = URLQualityScorerMLP()

    if os.path.exists('url_quality_scorer.pth'):
        print("\nLoading trained weights from url_quality_scorer.pth")
        model.load_state_dict(torch.load('url_quality_scorer.pth'))
    else:
        print("\nNo trained weights found, using random initialization")

    # Export to Jolt-compatible ONNX
    print("\nExporting to Jolt-compatible format...")
    export_jolt_compatible_onnx(model, 'network.onnx')

    # Test
    print("\nVerifying ONNX output...")
    test_onnx_model(model, 'network.onnx')

    # Update metadata
    meta = {
        "model_type": "url_quality_scorer",
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": NUM_CLASSES,
        "architecture": f"{INPUT_SIZE} -> {HIDDEN_SIZE} -> {NUM_CLASSES} MLP",
        "format": "jolt_atlas_compatible",
        "ops": ["MatMul", "Add", "Relu"],
        "weight_layout": "transposed",
        "classes": QUALITY_TIERS,
    }

    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("  - network.onnx: Jolt Atlas compatible")
    print("  - Uses MatMul + Add (not Gemm)")
    print("  - Weights transposed for correct matmul")
    print("=" * 60)
