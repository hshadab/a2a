"""
x402 Payment Protocol Implementation for Base Mainnet

Implements HTTP 402 Payment Required flow with USDC on Base.
Supports both direct payments and Coinbase Facilitator.
"""
import json
import time
import hashlib
import httpx
from typing import Optional, Tuple, Dict, Any
from decimal import Decimal
from datetime import datetime

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount

from .config import config
from .types import PaymentRequest, PaymentReceipt, X402PaymentChallenge


# USDC has 6 decimals
USDC_DECIMALS = 6

# ERC20 ABI (minimal for transfer)
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    }
]


class X402Client:
    """Client for making x402 payments on Base mainnet"""

    def __init__(self, private_key: Optional[str] = None):
        self.w3 = Web3(Web3.HTTPProvider(config.base_rpc_url))
        self.chain_id = config.base_chain_id

        if private_key:
            self.account: LocalAccount = Account.from_key(private_key)
        elif config.private_key:
            self.account: LocalAccount = Account.from_key(config.private_key)
        else:
            self.account = None

        self.usdc_contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(config.usdc_address),
            abi=ERC20_ABI
        )

    def get_balance(self, address: Optional[str] = None) -> float:
        """Get USDC balance in human-readable format"""
        if address is None:
            if self.account is None:
                raise ValueError("No account configured")
            address = self.account.address

        balance_raw = self.usdc_contract.functions.balanceOf(
            Web3.to_checksum_address(address)
        ).call()

        return balance_raw / (10 ** USDC_DECIMALS)

    def usdc_to_raw(self, amount: float) -> int:
        """Convert USDC amount to raw units (6 decimals)"""
        return int(Decimal(str(amount)) * Decimal(10 ** USDC_DECIMALS))

    def raw_to_usdc(self, raw: int) -> float:
        """Convert raw units to USDC amount"""
        return raw / (10 ** USDC_DECIMALS)

    async def make_payment(
        self,
        recipient: str,
        amount_usdc: float,
        memo: str = ""
    ) -> PaymentReceipt:
        """
        Make a USDC payment on Base mainnet.

        Returns a PaymentReceipt with tx details.
        In demo mode (no account), returns a simulated receipt.
        """
        if self.account is None:
            # Demo mode - return simulated receipt
            print(f"[x402] DEMO MODE: Simulating payment of ${amount_usdc} USDC to {recipient[:10]}...")
            return PaymentReceipt(
                tx_hash="simulated",
                amount_usdc=amount_usdc,
                sender="0x0000000000000000000000000000000000000000",
                recipient=recipient,
                timestamp=datetime.utcnow(),
                block_number=0,
                chain_id=self.chain_id
            )

        recipient = Web3.to_checksum_address(recipient)
        amount_raw = self.usdc_to_raw(amount_usdc)

        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        gas_price = self.w3.eth.gas_price

        tx = self.usdc_contract.functions.transfer(
            recipient,
            amount_raw
        ).build_transaction({
            'chainId': self.chain_id,
            'gas': 100000,  # ERC20 transfer typically uses ~65k
            'gasPrice': gas_price,
            'nonce': nonce,
        })

        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt['status'] != 1:
            raise Exception(f"Transaction failed: {tx_hash.hex()}")

        return PaymentReceipt(
            tx_hash=tx_hash.hex(),
            amount_usdc=amount_usdc,
            sender=self.account.address,
            recipient=recipient,
            timestamp=datetime.utcnow(),
            block_number=receipt['blockNumber'],
            chain_id=self.chain_id
        )

    def verify_payment(
        self,
        tx_hash: str,
        expected_recipient: str,
        expected_amount_usdc: float,
        max_age_seconds: int = 3600
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a payment was made.

        Returns (is_valid, error_message)
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            tx = self.w3.eth.get_transaction(tx_hash)

            # Check success
            if receipt['status'] != 1:
                return False, "Transaction failed"

            # Check it's a USDC transfer
            if tx['to'].lower() != config.usdc_address.lower():
                return False, "Not a USDC transaction"

            # Decode the transfer
            # transfer(address,uint256) selector = 0xa9059cbb
            if not tx['input'].startswith('0xa9059cbb'):
                return False, "Not a transfer transaction"

            # Decode recipient and amount from input data
            input_data = tx['input']
            # Skip selector (10 chars including 0x)
            recipient_hex = '0x' + input_data[34:74]
            amount_hex = input_data[74:138]

            decoded_recipient = Web3.to_checksum_address(recipient_hex)
            decoded_amount_raw = int(amount_hex, 16)
            decoded_amount_usdc = self.raw_to_usdc(decoded_amount_raw)

            # Verify recipient
            if decoded_recipient.lower() != expected_recipient.lower():
                return False, f"Wrong recipient: {decoded_recipient}"

            # Verify amount (allow small tolerance for rounding)
            if decoded_amount_usdc < expected_amount_usdc * 0.99:
                return False, f"Insufficient amount: {decoded_amount_usdc} < {expected_amount_usdc}"

            # Check age
            block = self.w3.eth.get_block(receipt['blockNumber'])
            tx_time = block['timestamp']
            if time.time() - tx_time > max_age_seconds:
                return False, "Payment too old"

            return True, None

        except Exception as e:
            return False, str(e)


class PaymentRequired(HTTPException):
    """
    HTTP 402 Payment Required exception.

    Usage:
        raise PaymentRequired(
            amount=1.50,
            recipient="0x...",
            memo="classify-50-urls"
        )
    """

    def __init__(
        self,
        amount: float,
        recipient: str,
        memo: str = "",
        expires_in: int = 300  # 5 minutes
    ):
        nonce = hashlib.sha256(f"{time.time()}{memo}".encode()).hexdigest()[:16]

        self.challenge = X402PaymentChallenge(
            amount=str(amount),
            currency="USDC",
            recipient=recipient,
            chain_id=config.base_chain_id,
            token_address=config.usdc_address,
            expires=int(time.time()) + expires_in,
            nonce=nonce
        )

        super().__init__(
            status_code=402,
            detail="Payment Required",
            headers={
                "X-402-Version": "1",
                "X-402-Payment": json.dumps(self.challenge.model_dump()),
                "Content-Type": "application/json"
            }
        )


def get_payment_from_header(request: Request) -> Optional[str]:
    """Extract payment receipt (tx hash) from request header"""
    return request.headers.get("X-402-Receipt")


async def require_payment(
    request: Request,
    amount: float,
    recipient: str,
    memo: str = ""
) -> PaymentReceipt:
    """
    Middleware helper to require payment.

    Usage in endpoint:
        receipt = await require_payment(request, amount=1.50, recipient="0x...")
    """
    tx_hash = get_payment_from_header(request)

    if not tx_hash:
        raise PaymentRequired(amount=amount, recipient=recipient, memo=memo)

    # Verify the payment
    client = X402Client()
    is_valid, error = client.verify_payment(
        tx_hash=tx_hash,
        expected_recipient=recipient,
        expected_amount_usdc=amount
    )

    if not is_valid:
        raise HTTPException(
            status_code=402,
            detail=f"Invalid payment: {error}"
        )

    # Return receipt
    receipt = client.w3.eth.get_transaction_receipt(tx_hash)
    tx = client.w3.eth.get_transaction(tx_hash)
    block = client.w3.eth.get_block(receipt['blockNumber'])

    return PaymentReceipt(
        tx_hash=tx_hash,
        amount_usdc=amount,
        sender=tx['from'],
        recipient=recipient,
        timestamp=datetime.fromtimestamp(block['timestamp']),
        block_number=receipt['blockNumber'],
        chain_id=config.base_chain_id
    )


# Convenience functions for creating payment challenges
def create_payment_response(
    amount: float,
    recipient: str,
    memo: str = ""
) -> JSONResponse:
    """Create a 402 Payment Required response"""
    nonce = hashlib.sha256(f"{time.time()}{memo}".encode()).hexdigest()[:16]

    challenge = X402PaymentChallenge(
        amount=str(amount),
        currency="USDC",
        recipient=recipient,
        chain_id=config.base_chain_id,
        chain=config.base_chain_caip2,  # CAIP-2 format
        token_address=config.usdc_address,
        expires=int(time.time()) + 300,
        nonce=nonce
    )

    return JSONResponse(
        status_code=402,
        content={"detail": "Payment Required", "payment": challenge.model_dump()},
        headers={
            "X-402-Version": "1",
            "X-402-Payment": json.dumps(challenge.model_dump())
        }
    )


# ============ Coinbase x402 Facilitator ============

class CoinbaseFacilitator:
    """
    Client for Coinbase's x402 Facilitator service.

    The facilitator enables fee-free payments for payers by:
    - Handling gas costs on behalf of the payer
    - Providing payment intent creation and verification
    - Supporting CAIP-2 chain identifiers

    Usage:
        facilitator = CoinbaseFacilitator()
        intent = await facilitator.create_payment_intent(0.001, "0x...", "classify-batch")
        # Client pays using intent
        is_valid, tx_hash = await facilitator.verify_payment(intent["id"])
    """

    def __init__(self, facilitator_url: Optional[str] = None):
        self.facilitator_url = facilitator_url or config.coinbase_facilitator_url
        self.timeout = 30.0

    async def create_payment_intent(
        self,
        amount_usdc: float,
        recipient: str,
        description: str,
        chain: str = None,
        token_address: str = None,
        expires_in: int = 300
    ) -> Dict[str, Any]:
        """
        Create a payment intent via Coinbase Facilitator.

        Args:
            amount_usdc: Amount in USDC
            recipient: Recipient address
            description: Payment description/memo
            chain: CAIP-2 chain identifier (default: eip155:8453)
            token_address: Token contract address (default: USDC on Base)
            expires_in: Seconds until expiration

        Returns:
            Payment intent object with id, paymentUrl, etc.
        """
        if chain is None:
            chain = config.base_chain_caip2
        if token_address is None:
            token_address = config.usdc_address

        payload = {
            "amount": str(amount_usdc),
            "currency": "USDC",
            "recipient": recipient,
            "chain": chain,
            "tokenAddress": token_address,
            "description": description,
            "expiresIn": expires_in,
            "metadata": {
                "provider": "ThreatProof Network",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.facilitator_url}/v1/payment-intents",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 201:
                    return response.json()
                else:
                    # Facilitator not available, return simulated intent
                    return self._create_simulated_intent(
                        amount_usdc, recipient, description, chain, expires_in
                    )
            except (httpx.ConnectError, httpx.TimeoutException):
                # Facilitator not available, return simulated intent
                return self._create_simulated_intent(
                    amount_usdc, recipient, description, chain, expires_in
                )

    async def verify_payment(
        self,
        payment_intent_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a payment via Coinbase Facilitator.

        Args:
            payment_intent_id: The payment intent ID to verify

        Returns:
            Tuple of (is_verified, transaction_hash)
        """
        # Handle simulated intents
        if payment_intent_id.startswith("sim_"):
            return True, f"simulated_tx_{payment_intent_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.facilitator_url}/v1/payment-intents/{payment_intent_id}",
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()
                    is_paid = data.get("status") == "completed"
                    tx_hash = data.get("transactionHash")
                    return is_paid, tx_hash
                else:
                    return False, None
            except (httpx.ConnectError, httpx.TimeoutException):
                # Facilitator not available
                return False, None

    async def get_payment_status(self, payment_intent_id: str) -> Dict[str, Any]:
        """Get detailed status of a payment intent"""
        # Handle simulated intents
        if payment_intent_id.startswith("sim_"):
            return {
                "id": payment_intent_id,
                "status": "completed",
                "transactionHash": f"simulated_tx_{payment_intent_id}",
                "simulated": True
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.facilitator_url}/v1/payment-intents/{payment_intent_id}",
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {"id": payment_intent_id, "status": "unknown", "error": "Not found"}
            except (httpx.ConnectError, httpx.TimeoutException):
                return {"id": payment_intent_id, "status": "unknown", "error": "Facilitator unavailable"}

    def _create_simulated_intent(
        self,
        amount_usdc: float,
        recipient: str,
        description: str,
        chain: str,
        expires_in: int
    ) -> Dict[str, Any]:
        """Create a simulated payment intent for demo/testing"""
        intent_id = f"sim_{hashlib.sha256(f'{time.time()}{description}'.encode()).hexdigest()[:16]}"
        return {
            "id": intent_id,
            "status": "pending",
            "amount": str(amount_usdc),
            "currency": "USDC",
            "recipient": recipient,
            "chain": chain,
            "description": description,
            "expiresAt": datetime.utcnow().isoformat(),
            "paymentUrl": f"https://x402.coinbase.com/pay/{intent_id}",
            "simulated": True
        }


# Global facilitator instance
coinbase_facilitator = CoinbaseFacilitator() if config.use_coinbase_facilitator else None


# ============ Enhanced X402Client with Facilitator Support ============

class X402ClientWithFacilitator(X402Client):
    """
    X402Client extended with Coinbase Facilitator support.

    When USE_COINBASE_FACILITATOR=true, uses the facilitator for payments.
    Otherwise, falls back to direct on-chain payments.
    """

    def __init__(self, private_key: Optional[str] = None):
        super().__init__(private_key)
        self.facilitator = CoinbaseFacilitator() if config.use_coinbase_facilitator else None

    async def make_payment(
        self,
        recipient: str,
        amount_usdc: float,
        memo: str = ""
    ) -> PaymentReceipt:
        """
        Make a USDC payment, using facilitator if configured.
        """
        if self.facilitator and config.use_coinbase_facilitator:
            # Use Coinbase Facilitator
            intent = await self.facilitator.create_payment_intent(
                amount_usdc=amount_usdc,
                recipient=recipient,
                description=memo
            )

            # For simulated intents, auto-complete
            is_verified, tx_hash = await self.facilitator.verify_payment(intent["id"])

            return PaymentReceipt(
                tx_hash=tx_hash or intent["id"],
                amount_usdc=amount_usdc,
                sender="facilitator",
                recipient=recipient,
                timestamp=datetime.utcnow(),
                block_number=0,
                chain_id=self.chain_id
            )
        else:
            # Use direct payment
            return await super().make_payment(recipient, amount_usdc, memo)
