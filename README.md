# CUDA Lennard-Jones Particle Simulator with Mimir Visualization

A high-performance CUDA-accelerated Lennard-Jones molecular dynamics simulator with real-time 3D visualization using the Mimir library.

## Features

- **GPU-accelerated simulation**: CUDA kernels for force calculation and integration
- **Real-time visualization**: Interactive 3D rendering using Mimir/Vulkan
- **Multiple initial configurations**: Random, crystalline lattice, and spherical shell
- **Benchmarking capabilities**: Performance measurement and analysis
- **Flexible parameters**: Adjustable simulation parameters and particle counts

## Project Structure

```
lj_simulator/
├── src/
│   ├── main.cpp              # Main application entry point
│   ├── lj_simulation.cu      # CUDA LJ simulation implementation
│   ├── particle_generator.cpp # Particle configuration generator
│   ├── particle_generator.hpp # Particle generator header
│   ├── benchmark.cpp         # Benchmarking implementation
│   └── benchmark.hpp         # Benchmarking header
├── include/
│   ├── lj_simulation.cuh     # LJ simulation header
│   └── validation.hpp        # CUDA error checking utilities
├── CMakeLists.txt           # CMake build configuration
├── batch_run.sh             # Batch benchmark script
└── README.md               # This file
```

## Dependencies

### System Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **CUDA SDK**: Version 10.0 or higher
- **Vulkan SDK**: Version 1.2 or higher
- **CMake**: Version 3.24 or higher
- **C++ Compiler**: C++20 compatible (GCC 10+ or Clang 10+)

### Libraries

The following libraries are automatically downloaded by CMake via FetchContent:

- **Mimir**: GPU visualization library
- **Slang**: Shading language
- **ImGui**: Immediate mode GUI
- **GLFW**: Window management
- **GLM**: Mathematics library

## Installation Instructions

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install essential build tools
sudo apt install build-essential cmake git

# Install CUDA Toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads
# Or use package manager:
sudo apt install nvidia-cuda-toolkit

# Install Vulkan SDK
sudo apt install vulkan-tools libvulkan-dev vulkan-utility-libraries-dev spirv-tools

# Install additional dependencies for GLFW
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
```

#### Arch Linux:
```bash
# Install dependencies
sudo pacman -S base-devel cmake git cuda vulkan-devel vulkan-validation-layers glfw-wayland libx11 libxrandr libxinerama libxcursor libxi mesa

# If using GLM from system packages, you may need to override FetchContent:
# Set FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER when configuring CMake
```

### 2. Install Mimir Library

```bash
# Clone Mimir repository
git clone https://github.com/temporal-hpc/mimir.git
cd mimir

# Build Mimir
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Install Mimir
sudo cmake --install . --prefix /usr/local

# Or install to custom location:
# cmake --install . --prefix $HOME/mimir_install
```

### 3. Verify CUDA Installation

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify CUDA samples work (optional)
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

### 4. Verify Vulkan Installation

```bash
# Check Vulkan installation
vulkaninfo | head -20

# Test Vulkan
vkcube  # Should display a rotating cube
```

## Building the Project

### 1. Clone and Setup

```bash
# Clone your project (if from repository) or create project directory
mkdir lj_simulator && cd lj_simulator

# Copy all source files to appropriate directories:
# - Copy CMakeLists.txt to root
# - Copy src/ files to src/
# - Copy include/ files to include/
# - Copy batch_run.sh to root
```

### 2. Configure and Build

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# If Mimir was installed to custom location:
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/mimir/install

# If having issues with GLM on Arch Linux:
# cmake .. -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER

# Build the project
cmake --build . -j$(nproc)
```

### 3. Verify Build

```bash
# Check if executables were created
ls -la bin/
# Should show: lj_simulator and lj_benchmark

# Test basic functionality
./bin/lj_simulator 1000 100 0 0  # 1000 particles, 100 steps, no viz, no benchmark
```

## Usage

### Basic Simulation

```bash
# Run with visualization (default parameters)
./bin/lj_simulator

# Run with custom parameters
./bin/lj_simulator [num_particles] [max_steps] [enable_viz] [enable_benchmark]

# Examples:
./bin/lj_simulator 5000 1000 1 0    # 5000 particles, 1000 steps, with visualization
./bin/lj_simulator 10000 5000 0 1   # 10000 particles, 5000 steps, headless with benchmark
```

### Benchmark Mode

```bash
# Run standalone benchmark
./bin/lj_benchmark [num_particles] [num_steps] [enable_viz]

# Batch benchmarking
chmod +x batch_run.sh
./batch_run.sh benchmark_results.csv
```

### Visualization Controls

When running with visualization enabled:

- **Ctrl+G**: Toggle control panel
- **Ctrl+Q**: Close window
- **Mouse**: Rotate camera view
- **Mouse wheel**: Zoom in/out

## Troubleshooting

### Common Issues

1. **CUDA not found**:
   ```bash
   export CUDA_ROOT=/usr/local/cuda
   export PATH=$PATH:/usr/local/cuda/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
   ```

2. **Vulkan validation layers missing**:
   ```bash
   sudo apt install vulkan-validationlayers-dev
   ```

3. **Mimir not found**:
   ```bash
   # Make sure CMAKE_PREFIX_PATH includes Mimir installation directory
   cmake .. -DCMAKE_PREFIX_PATH=/usr/local:/path/to/mimir/install
   ```

4. **GLM conflicts on Arch Linux**:
   ```bash
   cmake .. -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
   ```

5. **GPU memory issues**:
   - Reduce number of particles
   - Check available GPU memory with `nvidia-smi`

### Debug Build

For debugging issues:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DMIMIR_ENABLE_ASAN=ON
export ASAN_OPTIONS=protect_shadow_gap=0
./bin/lj_simulator
```

## Performance Tips

1. **Optimal particle counts**: Start with 1000-10000 particles for testing
2. **GPU architecture**: Ensure CUDA_ARCHITECTURES in CMakeLists.txt matches your GPU
3. **Block size**: The default 256 threads per block works well for most GPUs
4. **Cutoff distance**: Default 2.5σ balances accuracy and performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Mimir n-body sample from https://github.com/temporal-hpc/mimir
- Uses CUDA samples and best practices from NVIDIA
- Inspired by classical molecular dynamics simulation techniques