#!/usr/bin/env python3
"""
Test script for x402 payment flow on Base mainnet.

This script tests:
1. Wallet configuration
2. USDC balance check
3. Payment to treasury address
4. Receipt verification
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import config
from shared.x402 import X402Client
from shared.types import PaymentReceipt


async def test_wallet_config():
    """Test that wallet is properly configured."""
    print("\n=== Testing Wallet Configuration ===")

    if not config.private_key:
        print("❌ PRIVATE_KEY not set in environment")
        return False

    print(f"✓ Private key configured")
    print(f"  Chain ID: {config.base_chain_id}")
    print(f"  RPC URL: {config.base_rpc_url}")
    print(f"  USDC Address: {config.usdc_address}")
    print(f"  Treasury: {config.treasury_address}")

    return True


async def test_x402_client():
    """Test X402 client initialization and balance check."""
    print("\n=== Testing X402 Client ===")

    try:
        client = X402Client()
        print(f"✓ X402 client initialized")
        print(f"  Wallet address: {client.account.address}")

        # Check ETH balance for gas
        eth_balance = client.w3.eth.get_balance(client.account.address)
        eth_balance_formatted = client.w3.from_wei(eth_balance, 'ether')
        print(f"  ETH balance: {eth_balance_formatted} ETH")

        if eth_balance == 0:
            print("⚠️  Warning: No ETH for gas fees")

        # Check USDC balance
        usdc_balance = client.usdc_contract.functions.balanceOf(client.account.address).call()
        usdc_balance_formatted = usdc_balance / 1e6  # USDC has 6 decimals
        print(f"  USDC balance: ${usdc_balance_formatted:.6f}")

        if usdc_balance == 0:
            print("⚠️  Warning: No USDC balance")

        return True

    except Exception as e:
        print(f"❌ Error initializing X402 client: {e}")
        return False


async def test_payment_simulation():
    """Test payment flow (simulation only - no actual transfer)."""
    print("\n=== Testing Payment Flow (Simulation) ===")

    try:
        client = X402Client()

        # Calculate test payment
        test_amount = config.analyst_price_per_url  # $0.0005
        amount_raw = int(test_amount * 1e6)  # Convert to USDC units

        print(f"  Test amount: ${test_amount}")
        print(f"  Raw units: {amount_raw}")
        print(f"  Recipient: {config.treasury_address}")

        # Check if we have enough balance
        usdc_balance = client.usdc_contract.functions.balanceOf(client.account.address).call()

        if usdc_balance >= amount_raw:
            print(f"✓ Sufficient balance for test payment")
        else:
            print(f"⚠️  Insufficient balance (need {amount_raw}, have {usdc_balance})")

        # Build transaction without sending (dry run)
        nonce = client.w3.eth.get_transaction_count(client.account.address)
        gas_price = client.w3.eth.gas_price

        tx = client.usdc_contract.functions.transfer(
            config.treasury_address,
            amount_raw
        ).build_transaction({
            'chainId': config.base_chain_id,
            'gas': 100000,
            'gasPrice': gas_price,
            'nonce': nonce,
        })

        print(f"✓ Transaction built successfully (dry run)")
        print(f"  Gas limit: {tx['gas']}")
        print(f"  Gas price: {client.w3.from_wei(gas_price, 'gwei'):.2f} gwei")

        estimated_gas_cost = gas_price * tx['gas']
        print(f"  Estimated gas cost: {client.w3.from_wei(estimated_gas_cost, 'ether'):.8f} ETH")

        return True

    except Exception as e:
        print(f"❌ Error in payment simulation: {e}")
        return False


async def test_actual_payment():
    """Actually make a small test payment."""
    print("\n=== Making Actual Test Payment ===")
    print("⚠️  This will transfer real USDC on Base mainnet!")

    response = input("Proceed with test payment? (yes/no): ")
    if response.lower() != 'yes':
        print("Skipped actual payment test")
        return True

    try:
        client = X402Client()

        # Make a small test payment
        test_amount = 0.001  # $0.001 USDC

        receipt = await client.make_payment(
            recipient=config.treasury_address,
            amount_usdc=test_amount,
            memo="x402 test payment"
        )

        print(f"✓ Payment successful!")
        print(f"  Transaction hash: {receipt.tx_hash}")
        print(f"  Amount: ${receipt.amount_usdc}")
        print(f"  Block: {receipt.block_number}")
        print(f"  Timestamp: {receipt.timestamp}")

        # Verify the payment
        is_valid = await client.verify_payment(
            tx_hash=receipt.tx_hash,
            expected_recipient=config.treasury_address,
            expected_amount=test_amount
        )

        if is_valid:
            print(f"✓ Payment verified!")
        else:
            print(f"❌ Payment verification failed")

        return is_valid

    except Exception as e:
        print(f"❌ Error making payment: {e}")
        return False


async def main():
    print("=" * 50)
    print("  x402 Payment Flow Test")
    print("  Base Mainnet USDC")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Wallet Config", await test_wallet_config()))
    results.append(("X402 Client", await test_x402_client()))
    results.append(("Payment Simulation", await test_payment_simulation()))

    # Optional: actual payment
    if "--live" in sys.argv:
        results.append(("Actual Payment", await test_actual_payment()))

    # Summary
    print("\n" + "=" * 50)
    print("  Test Results")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check configuration.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
