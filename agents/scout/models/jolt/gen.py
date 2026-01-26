#!/usr/bin/env python3
"""
URL Quality Scorer Model for zkML Proof Generation

Creates a model that scores the quality/novelty of discovered URLs.
Scout uses this to prove it did actual work analyzing URLs before
presenting them to Analyst for classification.

Architecture designed to fit within Jolt Atlas MAX_TENSOR_SIZE=1024:
- Input: 32 features (URL quality indicators)
- Hidden: 16 neurons (32x16=512 weights, within limit)
- Output: 4 quality tiers (HIGH, MEDIUM, LOW, NOISE)

Features:
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
15: suspicious_keyword_count / 5
16: source_reputation (0-1)
17: is_novel (0/1) - not seen before
18: age_hours / 24 - how recently discovered
19: threat_feed_count / 5 - appears in how many feeds
20-31: additional context features + padding
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
INPUT_SIZE = 32      # URL quality features
HIDDEN_SIZE = 16     # 32x16=512 weights (within 1024)
NUM_CLASSES = 4      # HIGH, MEDIUM, LOW, NOISE
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Quality tier labels
QUALITY_TIERS = ["HIGH", "MEDIUM", "LOW", "NOISE"]


class URLQualityScorerMLP(nn.Module):
    """
    MLP scorer for URL quality/novelty assessment.
    Architecture: 32 -> 16 -> 4
    Max weight matrix: 32x16 = 512 (within MAX_TENSOR_SIZE=1024)
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super(URLQualityScorerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # 32 -> 16
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 16 -> 4
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)      # [batch, 16]
        x = self.relu(x)     # [batch, 16]
        x = self.fc2(x)      # [batch, 4]
        return x


def generate_synthetic_url_quality_features():
    """
    Generate synthetic URL quality features for training.
    Returns (features, label) where label is 0=HIGH, 1=MEDIUM, 2=LOW, 3=NOISE

    Quality assessment considers:
    - Novelty: Is this URL new/unseen?
    - Threat potential: Does it have phishing indicators?
    - Source reputation: Where did it come from?
    - Timeliness: How fresh is the discovery?
    """
    features = np.zeros(INPUT_SIZE, dtype=np.float32)

    # Decide quality tier first
    quality_type = random.choices(
        ['high', 'medium', 'low', 'noise'],
        weights=[0.20, 0.35, 0.30, 0.15]
    )[0]

    if quality_type == 'high':
        # HIGH quality: Novel, suspicious, from good source, recent
        features[0] = random.uniform(0.3, 0.8)    # url_length: moderate to long
        features[1] = random.uniform(0.3, 0.6)    # domain_length
        features[2] = random.uniform(0.2, 0.7)    # path_length
        features[3] = random.uniform(0.1, 0.5)    # query_length
        features[4] = random.uniform(0.2, 0.6)    # subdomain_count: some
        features[5] = random.choice([0.0, 1.0]) if random.random() < 0.2 else 0.0  # has_ip
        features[6] = random.choice([0.0, 1.0]) if random.random() < 0.1 else 0.0  # has_port
        features[7] = 0.0 if random.random() < 0.5 else 1.0  # uses_https: mixed
        features[8] = random.uniform(0.1, 0.5)    # digit_count
        features[9] = random.uniform(0.1, 0.4)    # special_char_count
        features[10] = random.uniform(0.1, 0.3)   # digit_ratio
        features[11] = random.uniform(0.5, 0.9)   # entropy: higher (suspicious)
        features[12] = random.uniform(0.5, 1.0)   # tld_risk_score: risky
        features[13] = random.uniform(0.4, 1.0)   # typosquat_score: high
        features[14] = 1.0 if random.random() < 0.6 else 0.0  # has_brand_match
        features[15] = random.uniform(0.3, 0.8)   # suspicious_keyword_count
        features[16] = random.uniform(0.7, 1.0)   # source_reputation: good
        features[17] = 1.0                         # is_novel: yes
        features[18] = random.uniform(0.0, 0.3)   # age_hours: very recent
        features[19] = random.uniform(0.0, 0.4)   # threat_feed_count: not everywhere
        # Additional features
        features[20] = random.uniform(0.5, 1.0)   # threat_potential
        features[21] = random.uniform(0.7, 1.0)   # discovery_priority
        label = 0  # HIGH

    elif quality_type == 'medium':
        # MEDIUM quality: Some novelty, moderate signals
        features[0] = random.uniform(0.2, 0.6)    # url_length
        features[1] = random.uniform(0.2, 0.5)    # domain_length
        features[2] = random.uniform(0.1, 0.5)    # path_length
        features[3] = random.uniform(0.0, 0.3)    # query_length
        features[4] = random.uniform(0.1, 0.4)    # subdomain_count
        features[5] = 0.0                          # has_ip: no
        features[6] = 0.0                          # has_port: no
        features[7] = 1.0 if random.random() < 0.6 else 0.0  # uses_https
        features[8] = random.uniform(0.05, 0.3)   # digit_count
        features[9] = random.uniform(0.05, 0.2)   # special_char_count
        features[10] = random.uniform(0.05, 0.2)  # digit_ratio
        features[11] = random.uniform(0.3, 0.6)   # entropy: moderate
        features[12] = random.uniform(0.2, 0.6)   # tld_risk_score
        features[13] = random.uniform(0.1, 0.5)   # typosquat_score
        features[14] = 1.0 if random.random() < 0.3 else 0.0  # has_brand_match
        features[15] = random.uniform(0.1, 0.4)   # suspicious_keyword_count
        features[16] = random.uniform(0.5, 0.8)   # source_reputation
        features[17] = 1.0 if random.random() < 0.7 else 0.0  # is_novel: probably
        features[18] = random.uniform(0.2, 0.6)   # age_hours: recent
        features[19] = random.uniform(0.2, 0.6)   # threat_feed_count
        features[20] = random.uniform(0.3, 0.6)   # threat_potential
        features[21] = random.uniform(0.4, 0.7)   # discovery_priority
        label = 1  # MEDIUM

    elif quality_type == 'low':
        # LOW quality: Already known, weak signals, older
        features[0] = random.uniform(0.1, 0.4)    # url_length: shorter
        features[1] = random.uniform(0.1, 0.4)    # domain_length
        features[2] = random.uniform(0.0, 0.3)    # path_length
        features[3] = random.uniform(0.0, 0.2)    # query_length
        features[4] = random.uniform(0.0, 0.2)    # subdomain_count
        features[5] = 0.0                          # has_ip: no
        features[6] = 0.0                          # has_port: no
        features[7] = 1.0 if random.random() < 0.8 else 0.0  # uses_https: usually
        features[8] = random.uniform(0.0, 0.2)    # digit_count
        features[9] = random.uniform(0.0, 0.1)    # special_char_count
        features[10] = random.uniform(0.0, 0.1)   # digit_ratio
        features[11] = random.uniform(0.2, 0.5)   # entropy: lower
        features[12] = random.uniform(0.1, 0.4)   # tld_risk_score
        features[13] = random.uniform(0.0, 0.3)   # typosquat_score: low
        features[14] = 0.0                         # has_brand_match
        features[15] = random.uniform(0.0, 0.2)   # suspicious_keyword_count
        features[16] = random.uniform(0.3, 0.6)   # source_reputation: moderate
        features[17] = 0.0                         # is_novel: no
        features[18] = random.uniform(0.5, 1.0)   # age_hours: older
        features[19] = random.uniform(0.5, 1.0)   # threat_feed_count: in many feeds
        features[20] = random.uniform(0.1, 0.4)   # threat_potential
        features[21] = random.uniform(0.2, 0.4)   # discovery_priority
        label = 2  # LOW

    else:  # noise
        # NOISE: Irrelevant, not threats, spam
        features[0] = random.uniform(0.0, 0.3)    # url_length: short
        features[1] = random.uniform(0.0, 0.2)    # domain_length: short
        features[2] = random.uniform(0.0, 0.2)    # path_length
        features[3] = random.uniform(0.0, 0.1)    # query_length
        features[4] = random.uniform(0.0, 0.1)    # subdomain_count: few
        features[5] = 0.0                          # has_ip: no
        features[6] = 0.0                          # has_port: no
        features[7] = 1.0                          # uses_https: yes (legitimate)
        features[8] = random.uniform(0.0, 0.1)    # digit_count: minimal
        features[9] = random.uniform(0.0, 0.05)   # special_char_count: minimal
        features[10] = random.uniform(0.0, 0.05)  # digit_ratio: very low
        features[11] = random.uniform(0.1, 0.3)   # entropy: low (normal)
        features[12] = random.uniform(0.0, 0.1)   # tld_risk_score: safe
        features[13] = random.uniform(0.0, 0.1)   # typosquat_score: none
        features[14] = 0.0                         # has_brand_match: no
        features[15] = random.uniform(0.0, 0.1)   # suspicious_keyword_count: none
        features[16] = random.uniform(0.1, 0.4)   # source_reputation: low
        features[17] = 0.0                         # is_novel: no (known safe)
        features[18] = random.uniform(0.8, 1.0)   # age_hours: old
        features[19] = random.uniform(0.0, 0.2)   # threat_feed_count: not in feeds
        features[20] = random.uniform(0.0, 0.1)   # threat_potential: none
        features[21] = random.uniform(0.0, 0.2)   # discovery_priority: low
        label = 3  # NOISE

    # Add some noise to all features
    noise = np.random.normal(0, 0.02, INPUT_SIZE).astype(np.float32)
    features = np.clip(features + noise, 0, 1)

    return features, label


def generate_dataset(num_samples=10000):
    """Generate synthetic URL quality dataset"""
    print(f"Generating {num_samples} synthetic URL quality samples...")

    X = []
    y = []

    for _ in range(num_samples):
        features, label = generate_synthetic_url_quality_features()
        X.append(features)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Print statistics
    high_count = np.sum(y == 0)
    medium_count = np.sum(y == 1)
    low_count = np.sum(y == 2)
    noise_count = np.sum(y == 3)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"HIGH: {high_count} ({high_count/num_samples*100:.1f}%)")
    print(f"MEDIUM: {medium_count} ({medium_count/num_samples*100:.1f}%)")
    print(f"LOW: {low_count} ({low_count/num_samples*100:.1f}%)")
    print(f"NOISE: {noise_count} ({noise_count/num_samples*100:.1f}%)")

    return X, y


def train_model(X, y):
    """Train the URL quality scorer model"""
    print("\n Training URL Quality Scorer Model")
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
    model = URLQualityScorerMLP()
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
            torch.save(model.state_dict(), 'url_quality_scorer.pth')

    print(f"\n Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model
    model.load_state_dict(torch.load('url_quality_scorer.pth'))

    # Calculate final training accuracy
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_predicted = torch.argmax(train_outputs, dim=1)
        train_acc = (train_predicted == y_train_tensor).float().mean().item() * 100

    return model, train_acc, best_val_acc


def test_sample_urls(model):
    """Test model with sample URL quality vectors"""
    print("\n Testing Sample URL Quality Scores")
    print("=" * 50)

    model.eval()

    # Manually crafted test cases
    test_cases = [
        {
            'name': 'High quality: Novel typosquat from good source',
            'features': create_high_quality_features(),
            'expected': 'HIGH'
        },
        {
            'name': 'Medium quality: Moderate signals, somewhat new',
            'features': create_medium_quality_features(),
            'expected': 'MEDIUM'
        },
        {
            'name': 'Low quality: Already known, old discovery',
            'features': create_low_quality_features(),
            'expected': 'LOW'
        },
        {
            'name': 'Noise: Legitimate site, no threat indicators',
            'features': create_noise_features(),
            'expected': 'NOISE'
        }
    ]

    for case in test_cases:
        features = torch.FloatTensor(case['features']).unsqueeze(0)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1).numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class] * 100

        prediction = QUALITY_TIERS[predicted_class]
        status = "OK" if prediction == case['expected'] else "XX"

        print(f"\n{status} {case['name']}")
        print(f"   Expected: {case['expected']}, Got: {prediction}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Probs: HIGH={probs[0]*100:.1f}%, MEDIUM={probs[1]*100:.1f}%, LOW={probs[2]*100:.1f}%, NOISE={probs[3]*100:.1f}%")


def create_high_quality_features():
    """Create a feature vector for a high quality URL discovery"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.6    # moderate URL length
    features[11] = 0.8   # high entropy
    features[12] = 0.9   # risky TLD
    features[13] = 0.9   # high typosquat score
    features[14] = 1.0   # brand match
    features[15] = 0.6   # suspicious keywords
    features[16] = 0.9   # high source reputation
    features[17] = 1.0   # is novel
    features[18] = 0.1   # very recent
    features[19] = 0.1   # not in many feeds yet
    features[20] = 0.9   # high threat potential
    features[21] = 0.9   # high discovery priority
    return features


def create_medium_quality_features():
    """Create a feature vector for a medium quality URL"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.4    # moderate URL length
    features[11] = 0.5   # moderate entropy
    features[12] = 0.4   # moderate TLD risk
    features[13] = 0.3   # some typosquat
    features[15] = 0.3   # some suspicious keywords
    features[16] = 0.6   # decent source reputation
    features[17] = 0.7   # probably novel
    features[18] = 0.4   # relatively recent
    features[19] = 0.4   # in some feeds
    features[20] = 0.5   # moderate threat potential
    features[21] = 0.5   # moderate priority
    return features


def create_low_quality_features():
    """Create a feature vector for a low quality URL"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.3    # shorter URL
    features[7] = 1.0    # uses HTTPS
    features[11] = 0.3   # low entropy
    features[12] = 0.2   # low TLD risk
    features[13] = 0.1   # low typosquat
    features[16] = 0.4   # moderate source reputation
    features[17] = 0.0   # not novel
    features[18] = 0.8   # old discovery
    features[19] = 0.8   # in many feeds already
    features[20] = 0.2   # low threat potential
    features[21] = 0.2   # low priority
    return features


def create_noise_features():
    """Create a feature vector for noise (legitimate site)"""
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    features[0] = 0.15   # short URL
    features[1] = 0.1    # short domain
    features[7] = 1.0    # uses HTTPS
    features[11] = 0.2   # low entropy
    features[12] = 0.05  # very safe TLD
    features[13] = 0.0   # no typosquat
    features[14] = 0.0   # no brand match
    features[15] = 0.0   # no suspicious keywords
    features[16] = 0.2   # low source reputation (random discovery)
    features[17] = 0.0   # not novel
    features[18] = 1.0   # very old
    features[19] = 0.0   # not in threat feeds
    features[20] = 0.0   # no threat potential
    features[21] = 0.0   # no priority
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

        print(f"OK Model exported and verified: {output_path}")
        return True

    except Exception as e:
        print(f"XX ONNX export failed: {e}")
        return False


def save_metadata(train_acc, val_acc):
    """Save model metadata"""

    # Vocab mapping for integer scaling
    vocab_mapping = {}
    feature_names = [
        'url_length', 'domain_length', 'path_length', 'query_length',
        'subdomain_count', 'has_ip_address', 'has_port', 'uses_https',
        'digit_count', 'special_char_count', 'digit_ratio', 'entropy',
        'tld_risk_score', 'typosquat_score', 'has_brand_match', 'suspicious_keyword_count',
        'source_reputation', 'is_novel', 'age_hours', 'threat_feed_count',
        'threat_potential', 'discovery_priority',
        'padding_22', 'padding_23', 'padding_24', 'padding_25',
        'padding_26', 'padding_27', 'padding_28', 'padding_29',
        'padding_30', 'padding_31'
    ]

    for i, name in enumerate(feature_names):
        vocab_mapping[name] = {"index": i, "feature_type": "quality_feature"}

    vocab_data = {
        'vocab_mapping': vocab_mapping,
        'feature_names': feature_names
    }

    with open('vocab.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)

    # Meta.json
    meta = {
        "model_type": "url_quality_scorer",
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": NUM_CLASSES,
        "architecture": f"{INPUT_SIZE} -> {HIDDEN_SIZE} -> {NUM_CLASSES} MLP",
        "max_tensor_size": INPUT_SIZE * HIDDEN_SIZE,
        "prover_compatible": True,
        "classes": QUALITY_TIERS,
        "training_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "purpose": "Score URL quality/novelty for Scout work proof"
    }

    with open('meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Labels.json
    with open('labels.json', 'w') as f:
        json.dump(QUALITY_TIERS, f, indent=2)

    print("OK Metadata saved to vocab.json, meta.json, labels.json")


if __name__ == '__main__':
    print(" URL Quality Scorer Model Generation")
    print("=" * 60)
    print(f"Architecture: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {NUM_CLASSES}")
    print(f"Max weight matrix: {INPUT_SIZE}x{HIDDEN_SIZE} = {INPUT_SIZE*HIDDEN_SIZE} (limit: 1024)")
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
    print(" URL Quality Scorer Model Complete!")
    print(f"   Input: {INPUT_SIZE} features (scaled floats 0-1)")
    print(f"   Architecture: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {NUM_CLASSES}")
    print(f"   Quality Tiers: {', '.join(QUALITY_TIERS)}")
    print(f"   Training Accuracy: {train_acc:.1f}%")
    print(f"   Validation Accuracy: {val_acc:.1f}%")
