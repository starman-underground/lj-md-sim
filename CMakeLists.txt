cmake_minimum_required(VERSION 3.24)
project(LJSimulator LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(mimir REQUIRED)
find_package(OpenMP)

# Add executable
add_executable(lj_simulator
    src/main.cpp
    src/lj_sim.cu
    src/particle_generator.cpp
)

# Set properties
set_target_properties(lj_simulator PROPERTIES
    CUDA_ARCHITECTURES "89"
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)

# Include directories
target_include_directories(lj_simulator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# CRITICAL: Correct linking order for static libraries
# Libraries that depend on others must come FIRST
target_link_libraries(lj_simulator PRIVATE
    mimir                    # Mimir first (depends on Slang)
    /opt/slang/lib/libslang.so     # Slang second (Mimir depends on this)
    /opt/slang/lib/libslang-glslang.so  # Slang-GLSLANG third
    CUDA::cudart                    # CUDA libraries last
    CUDA::cuda_driver
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lj_simulator PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler options
target_compile_options(lj_simulator PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

option(BUILD_BENCHMARKS "Build benchmark executables" ON)
if(BUILD_BENCHMARKS)
    add_executable(lj_benchmark
        src/benchmark.cpp
        src/lj_sim.cu
        src/particle_generator.cpp
    )
    
    set_target_properties(lj_benchmark PROPERTIES
        CUDA_ARCHITECTURES "89"
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
    )
    
    target_include_directories(lj_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${SLANG_INCLUDE_DIR}
    )
    
    target_link_libraries(lj_benchmark PRIVATE
        mimir                    # Mimir first (depends on Slang)
        /opt/slang/lib/libslang.so     # Slang second (Mimir depends on this)
        /opt/slang/lib/libslang-glslang.so  # Slang-GLSLANG third
        CUDA::cudart                    # CUDA libraries last
        CUDA::cuda_driver
    )
    
    if(SLANG_GLSLANG_LIBRARY)
        target_link_libraries(lj_benchmark PRIVATE ${SLANG_GLSLANG_LIBRARY})
    endif()
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(lj_benchmark PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    target_compile_options(lj_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
endif()