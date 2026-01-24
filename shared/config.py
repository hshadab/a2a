"""
Configuration for the Threat Intelligence Network
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/threat_intel")

    # Agent URLs
    scout_url: str = os.getenv("SCOUT_URL", "http://localhost:8000")
    policy_url: str = os.getenv("POLICY_URL", "http://localhost:8001")
    analyst_url: str = os.getenv("ANALYST_URL", "http://localhost:8002")

    # Base mainnet x402 config
    base_rpc_url: str = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
    base_chain_id: int = 8453
    usdc_address: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base

    # Wallet (for payments)
    private_key: Optional[str] = os.getenv("PRIVATE_KEY")
    treasury_address: str = os.getenv("TREASURY_ADDRESS", "")

    # Pricing (in USDC, keeping low as requested)
    policy_price_per_decision: float = 0.001  # $0.001 per authorization
    analyst_price_per_url: float = 0.0005     # $0.0005 per URL classification

    # Scout settings
    scout_interval_seconds: int = 300  # 5 minutes
    scout_batch_size: int = 50

    # Jolt Atlas paths - use jolt subdirectory models for zkML proof generation
    # Note: In containers, PYTHONPATH is /opt/render/project/src
    jolt_atlas_path: str = os.getenv("JOLT_ATLAS_PATH", "./jolt-atlas")
    jolt_model_dir: str = os.getenv("JOLT_MODEL_DIR", "./agents/policy/models/jolt")
    authorization_model_path: str = os.getenv(
        "AUTH_MODEL_PATH",
        os.path.join(os.getenv("JOLT_MODEL_DIR", "./agents/policy/models/jolt"), "network.onnx")
    )
    # Classifier model path - use env var or default (relative to PYTHONPATH in container)
    classifier_model_path: str = os.getenv(
        "CLASSIFIER_MODEL_PATH",
        os.path.join(os.getenv("PYTHONPATH", "."), "agents/analyst/models/jolt/network.onnx")
    )

    # WebSocket for UI
    websocket_url: str = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws")

    # Redis for coordination
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")


config = Config()
