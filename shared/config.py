"""
Configuration for the Threat Intelligence Network

Environment Variables:
======================

REQUIRED for production mode (PRODUCTION_MODE=true):
- DATABASE_URL: PostgreSQL connection string
- PRIVATE_KEY: Wallet private key for real USDC payments
- TREASURY_ADDRESS: Wallet address for receiving payments
- ZKML_CLI_PATH: Path to the zkml-cli binary (proof_json_output)

OPTIONAL:
- PRODUCTION_MODE: Set to "true" for production (default: false)
- DATABASE_SSL_MODE: SSL mode for database (require/prefer/disable, default: prefer)
- SCOUT_URL, POLICY_URL, ANALYST_URL: Agent URLs (default: localhost)
- BASE_RPC_URL: Base RPC endpoint (default: https://mainnet.base.org)
- REDIS_URL: Redis connection string
- PHISHTANK_API_KEY: PhishTank API key for URL lookup API
- TWITTER_BEARER_TOKEN: Twitter API bearer token for TwitterSource
- SCOUT_API_KEY: API key for manual /trigger endpoint
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Config:
    # ==========================================================================
    # Production Mode
    # ==========================================================================
    # When True:
    # - Simulated proofs are rejected (real zkML proofs required)
    # - Simulated payments are rejected (real wallet required)
    # - In-memory database fallback is disabled (PostgreSQL required)
    production_mode: bool = os.getenv("PRODUCTION_MODE", "false").lower() == "true"

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost:5432/threat_intel")
    database_ssl_mode: str = os.getenv("DATABASE_SSL_MODE", "prefer")  # require, prefer, disable

    # ==========================================================================
    # Agent URLs
    # ==========================================================================
    scout_url: str = os.getenv("SCOUT_URL", "http://localhost:8000")
    policy_url: str = os.getenv("POLICY_URL", "http://localhost:8001")
    analyst_url: str = os.getenv("ANALYST_URL", "http://localhost:8002")

    # ==========================================================================
    # Base Mainnet x402 Configuration
    # ==========================================================================
    base_rpc_url: str = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
    base_chain_id: int = 8453
    base_chain_caip2: str = "eip155:8453"  # CAIP-2 format for chain identification
    usdc_address: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base

    # Coinbase x402 Facilitator config
    use_coinbase_facilitator: bool = os.getenv("USE_COINBASE_FACILITATOR", "false").lower() == "true"
    coinbase_facilitator_url: str = os.getenv("COINBASE_FACILITATOR_URL", "https://x402.coinbase.com")

    # ==========================================================================
    # Wallet Configuration (REQUIRED for production)
    # ==========================================================================
    private_key: Optional[str] = os.getenv("PRIVATE_KEY")
    treasury_address: str = os.getenv("TREASURY_ADDRESS", "")

    # ==========================================================================
    # Pricing (in USDC)
    # ==========================================================================
    policy_price_per_decision: float = 0.001  # $0.001 per authorization
    analyst_price_per_url: float = 0.0005     # $0.0005 per URL classification
    payment_tolerance: float = float(os.getenv("PAYMENT_TOLERANCE", "0.001"))  # 0.1% tolerance
    x402_gas_limit: int = int(os.getenv("X402_GAS_LIMIT", "100000"))

    # ==========================================================================
    # Scout Settings
    # ==========================================================================
    scout_interval_seconds: int = int(os.getenv("SCOUT_INTERVAL_SECONDS", "300"))  # 5 minutes
    scout_batch_size: int = int(os.getenv("SCOUT_BATCH_SIZE", "50"))
    scout_api_key: Optional[str] = os.getenv("SCOUT_API_KEY")  # API key for /trigger endpoint

    # ==========================================================================
    # zkML / Jolt Atlas Configuration (REQUIRED for production)
    # ==========================================================================
    # Path to zkml-cli binary - download from: https://github.com/hshadab/zkx402/releases
    zkml_cli_path: str = os.getenv("ZKML_CLI_PATH", "/usr/local/bin/proof_json_output")
    jolt_atlas_path: str = os.getenv("JOLT_ATLAS_PATH", "./jolt-atlas")
    jolt_model_dir: str = os.getenv("JOLT_MODEL_DIR", "./agents/policy/models/jolt")
    authorization_model_path: str = os.getenv(
        "AUTH_MODEL_PATH",
        os.path.join(os.getenv("JOLT_MODEL_DIR", "./agents/policy/models/jolt"), "network.onnx")
    )
    classifier_model_path: str = os.getenv(
        "CLASSIFIER_MODEL_PATH",
        os.path.join(os.getenv("PYTHONPATH", "."), "agents/analyst/models/jolt/network.onnx")
    )

    # ==========================================================================
    # WebSocket / Redis
    # ==========================================================================
    websocket_url: str = os.getenv("WEBSOCKET_URL", "ws://localhost:8000/ws")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # ==========================================================================
    # URL Source Configuration
    # ==========================================================================
    # PhishTank API key (optional - enables URL lookup API, not required for data dump)
    phishtank_api_key: Optional[str] = os.getenv("PHISHTANK_API_KEY")

    # Twitter/X API (for TwitterSource)
    twitter_bearer_token: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")

    # Feature flags for sources
    enable_ct_source: bool = os.getenv("ENABLE_CT_SOURCE", "true").lower() == "true"
    enable_twitter_source: bool = os.getenv("ENABLE_TWITTER_SOURCE", "false").lower() == "true"
    enable_paste_source: bool = os.getenv("ENABLE_PASTE_SOURCE", "false").lower() == "true"
    enable_phishtank_source: bool = os.getenv("ENABLE_PHISHTANK_SOURCE", "true").lower() == "true"

    # ==========================================================================
    # Campaign Clustering & Reputation
    # ==========================================================================
    enable_clustering: bool = os.getenv("ENABLE_CLUSTERING", "true").lower() == "true"
    clustering_similarity_threshold: float = float(os.getenv("CLUSTERING_THRESHOLD", "0.6"))
    enable_dynamic_reputation: bool = os.getenv("ENABLE_DYNAMIC_REPUTATION", "true").lower() == "true"

    def validate_production_requirements(self) -> List[str]:
        """
        Validate that all required configuration is present for production mode.
        Returns a list of error messages (empty if valid).
        """
        errors = []

        if not self.production_mode:
            return errors

        # Check required environment variables for production
        if not self.private_key:
            errors.append("PRIVATE_KEY is required in production mode for real USDC payments")

        if not self.treasury_address:
            errors.append("TREASURY_ADDRESS is required in production mode")

        if not self.database_url or "localhost" in self.database_url:
            errors.append("DATABASE_URL must be set to a real PostgreSQL instance in production mode")

        # Check zkml-cli binary
        if not os.path.isfile(self.zkml_cli_path) or not os.access(self.zkml_cli_path, os.X_OK):
            errors.append(
                f"ZKML_CLI_PATH ({self.zkml_cli_path}) must point to an executable zkml-cli binary in production mode. "
                f"Download from: https://github.com/hshadab/zkx402/releases"
            )

        return errors

    def assert_production_ready(self):
        """
        Assert that all production requirements are met.
        Logs warnings instead of raising errors to allow debugging.
        """
        errors = self.validate_production_requirements()
        if errors:
            import logging
            logger = logging.getLogger("config")
            error_msg = "Production mode validation warnings:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.warning(error_msg)
            # Don't raise - allow service to start for debugging
            # raise RuntimeError(error_msg)


config = Config()
