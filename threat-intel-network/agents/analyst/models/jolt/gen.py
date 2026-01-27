#!/usr/bin/env python3
"""
URL Classifier Model for zkML Proof Generation

Creates a model that classifies URLs as PHISHING, SAFE, or SUSPICIOUS.
Architecture designed to fit within Jolt Atlas MAX_TENSOR_SIZE=1024:
- Input: 32 features (URL characteristics)
- Hidden: 16 neurons (32×16=512 weights, within limit)
- Output: 4 classes (16×4=64 weights, within limit)

Features from features.py:
0: url_length / 200
1: domain_length / 50
2: path_length / 100
3: query_length / 100
4: subdomain_count / 5
5: has_ip_address (0/1)
6: has_port (0/1)
7: uses_https (0/1)
8: digit_count / 20
9: special_char_count / 20
10: digit_ratio
11: entropy / 5
12: tld_risk_score
13: typosquat_score
14: has_brand_match (0/1)
15: levenshtein_distance / 10
16: path_depth / 5
17: query_param_count / 10
18: has_suspicious_path (0/1)
19: suspicious_keyword_count / 5
20-31: context features (domain_phish_rate, similar_domains_phish_rate, etc.) + padding
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------
# Configuration (fits within MAX_TENSOR_SIZE=1024)
# --------------------------
INPUT_SIZE = 32      # URL features from features.py
HIDDEN_SIZE = 16     # 32×16=512 weights (within 1024)
NUM_CLASSES = 4      # PHISHING, SAFE, SUSPICIOUS, PADDING
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Class labels
CLASS_NAMES = ["PHISHING", "SAFE", "SUSPICIOUS", "UNKNOWN"]


class URLClassifierMLP(nn.Module):
    """
    MLP classifier for URL phishing detection.
    Architecture: 32 → 16 → 4
    Max weight matrix: 32×16 = 512 (within MAX_TENSOR_SIZE=1024)
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super(URLClassifierMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # 32 → 16
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 16 → 4
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)      # [batch, 16]
        x = self.relu(x)     # [batch, 16]
        x = self.fc2(x)      # [batch, 4]
        return x


def generate_synthetic_url_features():
    """
    Generate synthetic URL features mimicking real phishing/safe URLs.
    Returns (features, label) where label is 0=PHISHING, 1=SAFE, 2=SUSPICIOUS
    """
    features = np.zeros(INPUT_SIZE, dtype=np.float32)

    # Decide classification first based on random "type"
    url_type = random.choices(
        ['phishing', 'safe', 'suspicious'],
        weights=[0.35, 0.45, 0.20]
    )[0]

    if url_type == 'phishing':
        # Phishing URL characteristics
        features[0] = random.uniform(0.4, 1.0)    # url_length: longer
        features[1] = random.uniform(0.3, 0.8)    # domain_length: moderate to long
        features[2] = random.uniform(0.2, 0.8)    # path_length: varies
        features[3] = random.uniform(0.0, 0.5)    # query_length
        features[4] = random.uniform(0.2, 0.8)    # subdomain_count: more subdomains
        features[5] = random.choice([0.0, 1.0]) if random.random() < 0.15 else 0.0  # has_ip
        features[6] = random.choice([0.0, 1.0]) if random.random() < 0.1 else 0.0   # has_port
        features[7] = 0.0 if random.random() < 0.6 else 1.0  # uses_https: often no
        features[8] = random.uniform(0.2, 0.6)    # digit_count: more digits
        features[9] = random.uniform(0.1, 0.5)    # special_char_count
        features[10] = random.uniform(0.1, 0.4)   # digit_ratio: higher
        features[11] = random.uniform(0.5, 1.0)   # entropy: higher (random-looking)
        features[12] = random.uniform(0.5, 1.0)   # tld_risk_score: risky TLDs
        features[13] = random.uniform(0.3, 1.0)   # typosquat_score: high
        features[14] = 1.0 if random.random() < 0.5 else 0.0  # has_brand_match
        features[15] = random.uniform(0.0, 0.3)   # levenshtein_distance: small (typosquat)
        features[16] = random.uniform(0.2, 0.8)   # path_depth
        features[17] = random.uniform(0.1, 0.5)   # query_param_count
        features[18] = 1.0 if random.random() < 0.7 else 0.0  # has_suspicious_path
        features[19] = random.uniform(0.2, 1.0)   # suspicious_keyword_count
        # Context features (simulated historical data)
        features[20] = random.uniform(0.3, 0.9)   # domain_phish_rate
        features[21] = random.uniform(0.2, 0.7)   # similar_domains_phish_rate
        features[22] = random.uniform(0.2, 0.6)   # registrar_phish_rate
        features[23] = random.uniform(0.1, 0.5)   # ip_phish_rate
        label = 0  # PHISHING

    elif url_type == 'safe':
        # Safe URL characteristics
        features[0] = random.uniform(0.1, 0.4)    # url_length: shorter
        features[1] = random.uniform(0.1, 0.4)    # domain_length: shorter
        features[2] = random.uniform(0.0, 0.3)    # path_length: shorter
        features[3] = random.uniform(0.0, 0.2)    # query_length: minimal
        features[4] = random.uniform(0.0, 0.2)    # subdomain_count: few
        features[5] = 0.0                          # has_ip: no
        features[6] = 0.0                          # has_port: no
        features[7] = 1.0 if random.random() < 0.9 else 0.0  # uses_https: yes
        features[8] = random.uniform(0.0, 0.2)    # digit_count: few
        features[9] = random.uniform(0.0, 0.1)    # special_char_count: minimal
        features[10] = random.uniform(0.0, 0.1)   # digit_ratio: low
        features[11] = random.uniform(0.2, 0.5)   # entropy: lower (readable)
        features[12] = random.uniform(0.0, 0.2)   # tld_risk_score: safe TLDs
        features[13] = random.uniform(0.0, 0.2)   # typosquat_score: low
        features[14] = 0.0                         # has_brand_match: legitimate
        features[15] = random.uniform(0.8, 1.0)   # levenshtein_distance: high (not typosquat)
        features[16] = random.uniform(0.0, 0.3)   # path_depth: shallow
        features[17] = random.uniform(0.0, 0.2)   # query_param_count: few
        features[18] = 0.0                         # has_suspicious_path: no
        features[19] = 0.0                         # suspicious_keyword_count: none
        # Context features
        features[20] = random.uniform(0.0, 0.1)   # domain_phish_rate: low
        features[21] = random.uniform(0.0, 0.1)   # similar_domains_phish_rate
        features[22] = random.uniform(0.0, 0.1)   # registrar_phish_rate
        features[23] = random.uniform(0.0, 0.1)   # ip_phish_rate
        label = 1  # SAFE

    else:  # suspicious
        # Suspicious URL characteristics (ambiguous)
        features[0] = random.uniform(0.2, 0.6)    # url_length: moderate
        features[1] = random.uniform(0.2, 0.5)    # domain_length
        features[2] = random.uniform(0.1, 0.5)    # path_length
        features[3] = random.uniform(0.1, 0.4)    # query_length
        features[4] = random.uniform(0.1, 0.4)    # subdomain_count
        features[5] = 0.0 if random.random() < 0.95 else 1.0
        features[6] = 0.0 if random.random() < 0.95 else 1.0
        features[7] = 1.0 if random.random() < 0.5 else 0.0  # uses_https: mixed
        features[8] = random.uniform(0.1, 0.4)    # digit_count
        features[9] = random.uniform(0.05, 0.3)   # special_char_count
        features[10] = random.uniform(0.05, 0.2)  # digit_ratio
        features[11] = random.uniform(0.3, 0.7)   # entropy: moderate
        features[12] = random.uniform(0.2, 0.6)   # tld_risk_score: mixed
        features[13] = random.uniform(0.1, 0.5)   # typosquat_score: some
        features[14] = 1.0 if random.random() < 0.3 else 0.0
        features[15] = random.uniform(0.3, 0.7)   # levenshtein_distance
        features[16] = random.uniform(0.1, 0.5)   # path_depth
        features[17] = random.uniform(0.1, 0.4)   # query_param_count
        features[18] = 1.0 if random.random() < 0.3 else 0.0  # has_suspicious_path
        features[19] = random.uniform(0.0, 0.4)   # suspicious_keyword_count
        # Context features
        features[20] = random.uniform(0.1, 0.5)   # domain_phish_rate
        features[21] = random.uniform(0.1, 0.4)   # similar_domains_phish_rate
        features[22] = random.uniform(0.1, 0.3)   # registrar_phish_rate
        features[23] = random.uniform(0.1, 0.3)   # ip_phish_rate
        label = 2  # SUSPICIOUS

    # Add some noise to all features
    noise = np.random.normal(0, 0.02, INPUT_SIZE).astype(np.float32)
    features = np.clip(features + noise, 0, 1)

    return features, label


def generate_dataset(num_samples=10000):
    """Generate synthetic URL classification dataset"""
    print(f"Generating {num_samples} synthetic URL samples...")

    X = []
    y = []

    for _ in range(num_samples):
        features, label = generate_synthetic_url_features()
        X.append(features)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Print statistics
    phishing_count = np.sum(y == 0)
    safe_count = np.sum(y == 1)
    suspicious_count = np.sum(y == 2)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"PHISHING: {phishing_count} ({phishing_count/num_samples*100:.1f}%)")
    print(f"SAFE: {safe_count} ({safe_count/num_samples*100:.1f}%)")
    print(f"SUSPICIOUS: {suspicious_count} ({suspicious_count/num_samples*100:.1f}%)")

    return X, y


def train_model(X, y):
    """Train the URL classifier model"""
    print("\n Training URL Classifier Model")
    print("=" * 50)

    # Split into train/validation
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model
    model = URLClassifierMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_predicted = torch.argmax(val_outputs, dim=1)
            val_acc = (val_predicted == y_val_tensor).float().mean().item() * 100

        train_loss /= len(train_loader)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'url_classifier.pth')

    print(f"\n Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model
    model.load_state_dict(torch.load('url_classifier.pth'))

    # Calculate final training accuracy
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_predicted = torch.argmax(train_outputs, dim=1)
        train_acc = (train_predicted == y_train_tensor).float().mean().item() * 100

    return model, train_acc, best_val_acc


def test_sample_urls(model):
    """Test model with sample URL feature vectors"""
    print("\n Testing Sample URL Classifications")
    print("=" * 50)

    model.eval()

    # Manually crafted test cases
    test_cases = [
        {
            'name': 'Obvious phishing (typosquat, suspicious path, risky TLD)',
            'features': create_phishing_features(),
            'expected': 'PHISHING'
        },
        {
            'name': 'Safe URL (HTTPS, known domain, clean path)',
            'features': create_safe_features(),
            'expected': 'SAFE'
        },
        {
            'name': 'Suspicious (mixed signals)',
            'features': create_suspicious_features(),
            'expected': 'SUSPICIOUS'
        }
    ]

    for case in test_cases:
        features = torch.FloatTensor(case['features']).unsqueeze(0)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1).numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class] * 100

        prediction = CLASS_NAMES[predicted_class]
        status = "✓" if prediction == case['expected'] else "✗"

        print(f"\n{status} {case['name']}")
        print(f"   Expected: {case['expected']}, Got: {prediction}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Probs: PHISHING={probs[0]*100:.1f}%, SAFE={probs[1]*100:.1f}%, SUSPICIOUS={probs[2]*100:.1f}%")


def create_phishing_features():
    """Create a feature vector for an obvious phishing URL"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.7    # long URL
    features[1] = 0.5    # moderate domain
    features[4] = 0.6    # multiple subdomains
    features[7] = 0.0    # no HTTPS
    features[10] = 0.3   # high digit ratio
    features[11] = 0.8   # high entropy
    features[12] = 0.9   # risky TLD
    features[13] = 0.9   # high typosquat score
    features[14] = 1.0   # brand match
    features[15] = 0.1   # low levenshtein (typosquat)
    features[18] = 1.0   # suspicious path
    features[19] = 0.8   # suspicious keywords
    features[20] = 0.7   # high domain phish rate
    return features


def create_safe_features():
    """Create a feature vector for a safe URL"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.2    # short URL
    features[1] = 0.2    # short domain
    features[4] = 0.1    # few subdomains
    features[7] = 1.0    # HTTPS
    features[10] = 0.05  # low digit ratio
    features[11] = 0.3   # low entropy
    features[12] = 0.1   # safe TLD
    features[13] = 0.0   # no typosquat
    features[14] = 0.0   # no brand match (is the real brand)
    features[15] = 1.0   # high levenshtein
    features[18] = 0.0   # no suspicious path
    features[19] = 0.0   # no suspicious keywords
    features[20] = 0.05  # low phish rate
    return features


def create_suspicious_features():
    """Create a feature vector for a suspicious URL"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.4    # moderate URL length
    features[1] = 0.35   # moderate domain
    features[4] = 0.3    # some subdomains
    features[7] = 0.0    # no HTTPS (suspicious)
    features[10] = 0.15  # moderate digit ratio
    features[11] = 0.5   # moderate entropy
    features[12] = 0.4   # moderate TLD risk
    features[13] = 0.3   # some typosquat similarity
    features[15] = 0.5   # moderate levenshtein
    features[18] = 0.0   # no suspicious path
    features[20] = 0.3   # moderate phish rate
    return features


def export_to_onnx(model, output_path='network.onnx'):
    """Export model to ONNX format"""
    print(f"\n Exporting model to ONNX: {output_path}")

    model.eval()
    dummy_input = torch.randn(1, INPUT_SIZE)

    try:
        import onnx

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        print(f"✓ Model exported and verified: {output_path}")
        return True

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False


def save_metadata(train_acc, val_acc):
    """Save model metadata"""

    # Vocab mapping for integer scaling (similar to policy model)
    vocab_mapping = {}
    feature_names = [
        'url_length', 'domain_length', 'path_length', 'query_length',
        'subdomain_count', 'has_ip_address', 'has_port', 'uses_https',
        'digit_count', 'special_char_count', 'digit_ratio', 'entropy',
        'tld_risk_score', 'typosquat_score', 'has_brand_match', 'levenshtein_distance',
        'path_depth', 'query_param_count', 'has_suspicious_path', 'suspicious_keyword_count',
        'domain_phish_rate', 'similar_domains_phish_rate', 'registrar_phish_rate', 'ip_phish_rate',
        'padding_24', 'padding_25', 'padding_26', 'padding_27',
        'padding_28', 'padding_29', 'padding_30', 'padding_31'
    ]

    for i, name in enumerate(feature_names):
        vocab_mapping[name] = {"index": i, "feature_type": "url_feature"}

    vocab_data = {
        'vocab_mapping': vocab_mapping,
        'feature_names': feature_names
    }

    with open('vocab.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)

    # Meta.json
    meta = {
        "model_type": "url_classifier",
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": NUM_CLASSES,
        "architecture": f"{INPUT_SIZE} -> {HIDDEN_SIZE} -> {NUM_CLASSES} MLP",
        "max_tensor_size": INPUT_SIZE * HIDDEN_SIZE,
        "prover_compatible": True,
        "classes": CLASS_NAMES,
        "training_accuracy": train_acc,
        "validation_accuracy": val_acc
    }

    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Labels.json
    with open('labels.json', 'w') as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print("✓ Metadata saved to vocab.json, meta.json, labels.json")


if __name__ == '__main__':
    print(" URL Classifier Model Generation")
    print("=" * 60)
    print(f"Architecture: {INPUT_SIZE} → {HIDDEN_SIZE} → {NUM_CLASSES}")
    print(f"Max weight matrix: {INPUT_SIZE}×{HIDDEN_SIZE} = {INPUT_SIZE*HIDDEN_SIZE} (limit: 1024)")
    print()

    # Generate dataset
    X, y = generate_dataset(num_samples=10000)

    # Train model
    model, train_acc, val_acc = train_model(X, y)

    # Test with samples
    test_sample_urls(model)

    # Export to ONNX
    export_success = export_to_onnx(model, 'network.onnx')

    # Save metadata
    save_metadata(train_acc, val_acc)

    print("\n" + "=" * 60)
    print(" URL Classifier Model Complete!")
    print(f"   Input: {INPUT_SIZE} features (scaled floats 0-1)")
    print(f"   Architecture: {INPUT_SIZE} → {HIDDEN_SIZE} → {NUM_CLASSES}")
    print(f"   Classes: {', '.join(CLASS_NAMES)}")
    print(f"   Training Accuracy: {train_acc:.1f}%")
    print(f"   Validation Accuracy: {val_acc:.1f}%")
