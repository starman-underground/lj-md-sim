#!/bin/bash
# save as verify_build.sh

echo "=== Build Verification Script ==="

# Check required files
REQUIRED_FILES=(
    "src/main.cpp"
    "src/lj_sim.cu" 
    "src/lj_cpu.cpp"
    "src/particle_generator.cpp"
    "src/benchmark.cpp"
    "include/lj_sim.cuh"
    "include/particle_generator.hpp"
    "include/benchmark.hpp"
    "include/validation.hpp"
)

echo "Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done

# Check Mimir installation
echo "Checking Mimir installation..."
if pkg-config --exists mimir; then
    echo "✓ Mimir found via pkg-config"
elif find /usr/local -name "*mimir*" -type f 2>/dev/null | grep -q .; then
    echo "✓ Mimir installation found"
else
    echo "⚠ Mimir may not be properly installed"
fi

# Check CUDA
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler found: $(nvcc --version | grep release)"
else
    echo "✗ CUDA compiler not found"
    exit 1
fi

# Check Vulkan
echo "Checking Vulkan..."
if command -v vulkaninfo &> /dev/null; then
    echo "✓ Vulkan found"
else
    echo "⚠ Vulkan tools not found"
fi

echo "=== Verification Complete ==="
