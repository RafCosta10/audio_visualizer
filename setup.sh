#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================"
echo " Setting up Retro Audio Visualizer..."
echo "========================================"

echo ""
echo "--> Updating apt package lists..."
sudo apt-get update

echo ""
echo "--> Installing system dependencies..."
# build-essential & pkg-config: Required to compile C bindings
# libasound2-dev: Required by `cpal` for ALSA audio capture
# libdbus-1-dev: Required by `mpris` to read media player metadata
# libfontconfig1-dev: Required for font rendering
# libx11-dev, libwayland-dev, libxkbcommon-dev, etc: Required by `eframe/winit` for window creation
sudo apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    libasound2-dev \
    libdbus-1-dev \
    libfontconfig1-dev \
    libx11-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libwayland-dev \
    libegl1-mesa-dev

echo ""
echo "--> Checking for Rust installation..."
if ! command -v cargo &> /dev/null
then
    echo "Rust could not be found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    # Source the cargo environment so it's available in this script execution
    source "$HOME/.cargo/env"
    echo "Rust installed successfully."
else
    echo "Rust is already installed."
fi

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo "If Rust was just installed, you may need to run:"
echo "  source \$HOME/.cargo/env"
echo "Or simply close and reopen your terminal."
echo ""
echo "You can now build and run your visualizer with:"
echo "  cargo run --release"
echo "========================================"
