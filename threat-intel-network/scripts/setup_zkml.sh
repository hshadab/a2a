#!/bin/bash
# Setup script for zkML (Jolt Atlas) prover binary
#
# Downloads the pre-built proof_json_output binary from GitHub releases
# Binary source: https://github.com/hshadab/zkx402/releases
#
# Usage:
#   ./scripts/setup_zkml.sh [install_path]
#
# Default install path: /usr/local/bin/proof_json_output

set -e

# Configuration
ZKML_RELEASE_URL="https://github.com/hshadab/zkx402/releases/download/jolt-binary-v1/proof_json_output"
DEFAULT_INSTALL_PATH="/usr/local/bin/proof_json_output"

# Parse arguments
INSTALL_PATH="${1:-$DEFAULT_INSTALL_PATH}"

echo "==================================================="
echo "  ThreatProof zkML Prover Setup"
echo "==================================================="
echo ""
echo "This script downloads the Jolt Atlas zkML prover binary."
echo "Binary size: ~143MB"
echo ""
echo "Install path: $INSTALL_PATH"
echo ""

# Check if already installed
if [ -f "$INSTALL_PATH" ] && [ -x "$INSTALL_PATH" ]; then
    echo "zkML prover already installed at $INSTALL_PATH"
    echo "To reinstall, delete the file first."
    exit 0
fi

# Check if we need sudo
NEEDS_SUDO=false
INSTALL_DIR=$(dirname "$INSTALL_PATH")
if [ ! -w "$INSTALL_DIR" ]; then
    NEEDS_SUDO=true
    echo "Note: Will need sudo to install to $INSTALL_DIR"
fi

# Download binary
echo ""
echo "Downloading zkML prover binary..."
TEMP_FILE=$(mktemp)

if command -v curl &> /dev/null; then
    curl -fSL --progress-bar -o "$TEMP_FILE" "$ZKML_RELEASE_URL"
elif command -v wget &> /dev/null; then
    wget -q --show-progress -O "$TEMP_FILE" "$ZKML_RELEASE_URL"
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Verify download
if [ ! -s "$TEMP_FILE" ]; then
    echo "Error: Download failed or file is empty"
    rm -f "$TEMP_FILE"
    exit 1
fi

echo ""
echo "Download complete. Installing..."

# Install binary
if [ "$NEEDS_SUDO" = true ]; then
    sudo mv "$TEMP_FILE" "$INSTALL_PATH"
    sudo chmod +x "$INSTALL_PATH"
else
    mv "$TEMP_FILE" "$INSTALL_PATH"
    chmod +x "$INSTALL_PATH"
fi

# Verify installation
if [ -x "$INSTALL_PATH" ]; then
    echo ""
    echo "==================================================="
    echo "  Installation successful!"
    echo "==================================================="
    echo ""
    echo "Binary installed at: $INSTALL_PATH"
    echo ""
    echo "Add this to your .env file:"
    echo "  ZKML_CLI_PATH=$INSTALL_PATH"
    echo ""
    echo "Or set as environment variable:"
    echo "  export ZKML_CLI_PATH=$INSTALL_PATH"
    echo ""
else
    echo "Error: Installation failed"
    exit 1
fi
