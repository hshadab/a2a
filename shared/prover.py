"""
Jolt Atlas zkML Prover Wrapper

Provides Python interface to Jolt Atlas for proof generation and verification.
Supports both real ONNX inference and simulated mode for development.
"""
import subprocess
import json
import hashlib
import time
import tempfile
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from .config import config
from .logging_config import prover_logger as logger

# Try to import ONNX runtime for real inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available, using simulated inference")


@dataclass
class ProofResult:
    """Result of proof generation"""
    proof: bytes
    proof_hex: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    output: Any
    prove_time_ms: int


@dataclass
class VerifyResult:
    """Result of proof verification"""
    valid: bool
    verify_time_ms: int
    error: Optional[str] = None


def compute_commitment(data: Any) -> str:
    """Compute a commitment (hash) for data"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    return hashlib.sha256(data_str.encode()).hexdigest()


class JoltAtlasProver:
    """
    Wrapper for Jolt Atlas zkML prover.

    Uses CLI interface to generate and verify proofs.
    """

    def __init__(self, jolt_path: Optional[str] = None):
        self.jolt_path = jolt_path or config.jolt_atlas_path
        self.binary_path = os.path.join(self.jolt_path, "target/release/jolt-atlas")

        # Model commitments (computed once from model files)
        self._model_commitments: Dict[str, str] = {}

    def get_model_commitment(self, model_path: str) -> str:
        """Get or compute model commitment"""
        if model_path not in self._model_commitments:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_bytes = f.read()
                self._model_commitments[model_path] = hashlib.sha256(model_bytes).hexdigest()
            else:
                # For demo/testing, use a placeholder
                self._model_commitments[model_path] = hashlib.sha256(model_path.encode()).hexdigest()
        return self._model_commitments[model_path]

    async def generate_proof(
        self,
        model_path: str,
        inputs: List[float],
        model_name: str = "model"
    ) -> ProofResult:
        """
        Generate a zkML proof for model inference.

        Args:
            model_path: Path to ONNX model
            inputs: Input features as list of floats
            model_name: Name for logging

        Returns:
            ProofResult with proof bytes and commitments
        """
        start_time = time.time()

        # Compute commitments
        model_commitment = self.get_model_commitment(model_path)
        input_commitment = compute_commitment(inputs)

        # Create temp files for input/output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"inputs": inputs}, f)
            input_file = f.name

        output_file = tempfile.mktemp(suffix='.json')
        proof_file = tempfile.mktemp(suffix='.proof')

        try:
            # Check if Jolt Atlas binary exists
            if os.path.exists(self.binary_path):
                # Run actual Jolt Atlas
                result = subprocess.run(
                    [
                        self.binary_path,
                        "prove",
                        "--model", model_path,
                        "--input", input_file,
                        "--output", output_file,
                        "--proof", proof_file
                    ],
                    capture_output=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode != 0:
                    raise Exception(f"Jolt Atlas error: {result.stderr.decode()}")

                # Read output and proof
                with open(output_file, 'r') as f:
                    output_data = json.load(f)

                with open(proof_file, 'rb') as f:
                    proof_bytes = f.read()

            else:
                # Simulated proof for development/demo
                # In production, Jolt Atlas must be installed
                output_data = self._simulate_inference(model_name, inputs)
                proof_bytes = self._simulate_proof(model_commitment, input_commitment, output_data)

            # Compute output commitment
            output_commitment = compute_commitment(output_data)

            prove_time = int((time.time() - start_time) * 1000)
            proof_hex = proof_bytes.hex()
            proof_hash = hashlib.sha256(proof_bytes).hexdigest()

            return ProofResult(
                proof=proof_bytes,
                proof_hex=proof_hex,
                proof_hash=proof_hash,
                model_commitment=model_commitment,
                input_commitment=input_commitment,
                output_commitment=output_commitment,
                output=output_data,
                prove_time_ms=prove_time
            )

        finally:
            # Cleanup temp files
            for f in [input_file, output_file, proof_file]:
                if os.path.exists(f):
                    os.remove(f)

    async def verify_proof(
        self,
        proof: bytes,
        model_commitment: str,
        input_commitment: str,
        output_commitment: str
    ) -> VerifyResult:
        """
        Verify a zkML proof.

        Args:
            proof: Proof bytes
            model_commitment: Expected model commitment
            input_commitment: Expected input commitment
            output_commitment: Expected output commitment

        Returns:
            VerifyResult with validity and timing
        """
        start_time = time.time()

        # Create temp files
        proof_file = tempfile.mktemp(suffix='.proof')
        public_inputs_file = tempfile.mktemp(suffix='.json')

        try:
            with open(proof_file, 'wb') as f:
                f.write(proof)

            public_inputs = {
                "model_commitment": model_commitment,
                "input_commitment": input_commitment,
                "output_commitment": output_commitment
            }
            with open(public_inputs_file, 'w') as f:
                json.dump(public_inputs, f)

            if os.path.exists(self.binary_path):
                # Run actual verification
                result = subprocess.run(
                    [
                        self.binary_path,
                        "verify",
                        "--proof", proof_file,
                        "--public-inputs", public_inputs_file
                    ],
                    capture_output=True,
                    timeout=60
                )

                valid = result.returncode == 0
                error = None if valid else result.stderr.decode()

            else:
                # Simulated verification for development
                valid, error = self._simulate_verification(
                    proof, model_commitment, input_commitment, output_commitment
                )

            verify_time = int((time.time() - start_time) * 1000)

            return VerifyResult(
                valid=valid,
                verify_time_ms=verify_time,
                error=error
            )

        finally:
            for f in [proof_file, public_inputs_file]:
                if os.path.exists(f):
                    os.remove(f)

    def _run_onnx_inference(self, model_path: str, inputs: List[float]) -> Optional[np.ndarray]:
        """Run real ONNX inference if available"""
        if not ONNX_AVAILABLE or not os.path.exists(model_path):
            return None

        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape

            # Reshape inputs to match model expectations
            expected_size = input_shape[1] if len(input_shape) > 1 else len(inputs)
            if len(inputs) < expected_size:
                inputs = inputs + [0.0] * (expected_size - len(inputs))
            elif len(inputs) > expected_size:
                inputs = inputs[:expected_size]

            input_array = np.array([inputs], dtype=np.float32)
            outputs = session.run(None, {input_name: input_array})
            logger.debug(f"ONNX inference completed for {model_path}")
            return outputs[0]
        except Exception as e:
            logger.warning(f"ONNX inference failed: {e}, falling back to simulation")
            return None

    def _simulate_inference(self, model_name: str, inputs: List[float]) -> Dict[str, Any]:
        """Run model inference - uses real ONNX when available, otherwise simulates"""

        # Try real ONNX inference first
        if "authorization" in model_name.lower():
            model_path = config.authorization_model_path
            outputs = self._run_onnx_inference(model_path, inputs)

            if outputs is not None:
                # Real ONNX output: [deny_score, approve_score]
                scores = outputs[0]
                decision = "AUTHORIZED" if scores[1] > scores[0] else "DENIED"
                confidence = float(max(scores))
                logger.info(f"Real ONNX authorization: {decision} (confidence: {confidence:.2f})")
                return {
                    "decision": decision,
                    "confidence": confidence,
                    "scores": scores.tolist()
                }

            # Fallback: simulated authorization
            cost_ratio = inputs[7] if len(inputs) > 7 else 0.1
            decision = "AUTHORIZED" if cost_ratio < 0.5 else "DENIED"
            confidence = min(0.99, max(0.7, 0.9 - cost_ratio))
            return {
                "decision": decision,
                "confidence": confidence,
                "scores": [1 - confidence, confidence]
            }
        else:
            # URL classifier
            model_path = config.classifier_model_path
            outputs = self._run_onnx_inference(model_path, inputs)

            if outputs is not None:
                # Real ONNX output: [phishing_score, safe_score, suspicious_score]
                scores = outputs[0]
                class_idx = int(np.argmax(scores))
                classifications = ["PHISHING", "SAFE", "SUSPICIOUS"]
                classification = classifications[class_idx]
                confidence = float(scores[class_idx])
                logger.info(f"Real ONNX classification: {classification} (confidence: {confidence:.2f})")
                return {
                    "classification": classification,
                    "confidence": confidence,
                    "scores": scores.tolist()
                }

            # Fallback: simulated classification
            risk_score = sum(inputs[:10]) / 10 if len(inputs) >= 10 else 0.5

            if risk_score > 0.7:
                classification = "PHISHING"
                confidence = min(0.99, risk_score)
            elif risk_score < 0.3:
                classification = "SAFE"
                confidence = min(0.99, 1 - risk_score)
            else:
                classification = "SUSPICIOUS"
                confidence = 0.6

            return {
                "classification": classification,
                "confidence": confidence,
                "scores": [risk_score, 1 - risk_score, 0.5 - abs(risk_score - 0.5)]
            }

    def _simulate_proof(
        self,
        model_commitment: str,
        input_commitment: str,
        output: Dict[str, Any]
    ) -> bytes:
        """Generate a simulated proof for development"""
        # Create a deterministic "proof" based on inputs
        proof_data = {
            "model_commitment": model_commitment,
            "input_commitment": input_commitment,
            "output_commitment": compute_commitment(output),
            "timestamp": int(time.time()),
            "simulated": True
        }
        proof_json = json.dumps(proof_data, sort_keys=True)
        # Add some padding to make it look like a real proof
        padded = proof_json + ("0" * 1000)
        return padded.encode()

    def _simulate_verification(
        self,
        proof: bytes,
        model_commitment: str,
        input_commitment: str,
        output_commitment: str
    ) -> Tuple[bool, Optional[str]]:
        """Simulate proof verification for development"""
        try:
            # Decode the simulated proof
            proof_str = proof.decode().rstrip("0")
            proof_data = json.loads(proof_str)

            # Check commitments match
            if proof_data.get("model_commitment") != model_commitment:
                return False, "Model commitment mismatch"

            if proof_data.get("input_commitment") != input_commitment:
                return False, "Input commitment mismatch"

            if proof_data.get("output_commitment") != output_commitment:
                return False, "Output commitment mismatch"

            return True, None

        except Exception as e:
            return False, str(e)


# ============ Authorization Prover ============

class AuthorizationProver:
    """
    Prover specifically for authorization decisions.
    """

    def __init__(self):
        self.prover = JoltAtlasProver()
        self.model_path = config.authorization_model_path

    async def prove_authorization(
        self,
        batch_size: int,
        budget_remaining: float,
        estimated_cost: float,
        source_reputation: float,
        novelty_score: float,
        time_since_last: int,
        threat_level: float
    ) -> ProofResult:
        """
        Generate proof for an authorization decision.

        The authorization model takes 64 features (padded if necessary).
        """
        # Encode inputs as features (normalized to 0-1 range)
        inputs = [
            batch_size / 100,                    # Normalized batch size
            budget_remaining / 10000,            # Normalized budget
            estimated_cost / 100,                # Normalized cost
            source_reputation,                    # Already 0-1
            novelty_score,                        # Already 0-1
            min(time_since_last / 3600, 1.0),    # Normalized time (max 1 hour)
            threat_level,                         # Already 0-1
            estimated_cost / max(budget_remaining, 0.01),  # Cost ratio
        ]

        # Pad to 64 features (as expected by Jolt Atlas authorization model)
        while len(inputs) < 64:
            inputs.append(0.0)

        return await self.prover.generate_proof(
            model_path=self.model_path,
            inputs=inputs,
            model_name="authorization"
        )

    async def verify_authorization(
        self,
        proof: bytes,
        model_commitment: str,
        input_commitment: str,
        output_commitment: str
    ) -> VerifyResult:
        """Verify an authorization proof"""
        return await self.prover.verify_proof(
            proof=proof,
            model_commitment=model_commitment,
            input_commitment=input_commitment,
            output_commitment=output_commitment
        )


# ============ URL Classifier Prover ============

class URLClassifierProver:
    """
    Prover for URL classification.
    """

    def __init__(self):
        self.prover = JoltAtlasProver()
        self.model_path = config.classifier_model_path

    async def prove_classification(self, features: List[float]) -> ProofResult:
        """
        Generate proof for a URL classification.

        Features should be extracted from the URL (see features.py).
        """
        # Ensure we have the right number of features
        # Pad or truncate to expected size
        expected_size = 32  # Adjust based on actual model
        if len(features) < expected_size:
            features = features + [0.0] * (expected_size - len(features))
        elif len(features) > expected_size:
            features = features[:expected_size]

        return await self.prover.generate_proof(
            model_path=self.model_path,
            inputs=features,
            model_name="url_classifier"
        )

    async def prove_batch_classification(
        self,
        all_features: List[List[float]]
    ) -> Tuple[List[ProofResult], ProofResult]:
        """
        Generate proofs for a batch of URL classifications.

        Returns individual proofs and a batch aggregation proof.
        """
        individual_proofs = []
        for features in all_features:
            proof = await self.prove_classification(features)
            individual_proofs.append(proof)

        # For batch proof, we create a commitment over all individual proofs
        batch_inputs = [
            float(int(p.proof_hash[:8], 16)) / (16**8)
            for p in individual_proofs
        ]

        batch_proof = await self.prover.generate_proof(
            model_path=self.model_path,
            inputs=batch_inputs[:32],  # Use first 32 for batch proof
            model_name="batch_classifier"
        )

        return individual_proofs, batch_proof

    async def verify_classification(
        self,
        proof: bytes,
        model_commitment: str,
        input_commitment: str,
        output_commitment: str
    ) -> VerifyResult:
        """Verify a classification proof"""
        return await self.prover.verify_proof(
            proof=proof,
            model_commitment=model_commitment,
            input_commitment=input_commitment,
            output_commitment=output_commitment
        )


# Global prover instances
authorization_prover = AuthorizationProver()
classifier_prover = URLClassifierProver()
