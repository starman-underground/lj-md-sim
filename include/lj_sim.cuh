#pragma once

#include <cuda_runtime.h>
#include <vector>

// CUDA error checking macro
#define checkCuda(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Lennard-Jones potential parameters
struct LJParams {
    float sigma;      // Characteristic length
    float epsilon;    // Energy well depth  
    float cutoff;     // Cutoff distance
    float cutoff_sq;  // Cutoff distance squared
};

// Device data structure for particles
struct ParticleData {
    float4* positions;  // x, y, z, mass
    float4* velocities; // vx, vy, vz, inv_mass
    float3* forces;     // fx, fy, fz
};

// Host data structure for particles
struct HostParticleData {
    std::vector<float4> positions;
    std::vector<float4> velocities;
    std::vector<float3> forces;
};

class LJSystem {
public:
    LJSystem(int num_particles, float sigma = 1.0f, float epsilon = 1.0f, float cutoff = 2.5f);
    ~LJSystem();
    
    // System setup
    void setPositions(const float4* host_positions);
    void setVelocities(const float4* host_velocities);
    void setLJParameters(float sigma, float epsilon, float cutoff);
    
    // Simulation step
    void step(float dt);
    
    // Energy calculations
    float getKineticEnergy();
    float getPotentialEnergy();
    
    // Data access
    void getPositions(float4* host_positions);
    void getVelocities(float4* host_velocities);
    
    // Public member for Mimir integration
    float4* d_positions;
    
private:
    int m_num_particles;
    int m_block_size;
    int m_grid_size;
    
    LJParams m_lj_params;
    
    // Device memory
    float4* d_velocities;
    float3* d_forces;
    
    // Helper methods
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void computeForces();
    void integrateMotion(float dt);
};

// CUDA kernels (implemented in .cu file)
extern "C" {
    void launch_lj_force_kernel(
        float4* positions,
        float3* forces,
        int num_particles,
        LJParams params,
        int block_size
    );
    
    void launch_integration_kernel(
        float4* positions,
        float4* velocities,
        float3* forces,
        int num_particles,
        float dt,
        int block_size
    );
    
    float launch_kinetic_energy_kernel(
        float4* velocities,
        int num_particles,
        int block_size
    );
    
    float launch_potential_energy_kernel(
        float4* positions,
        int num_particles,
        LJParams params,
        int block_size
    );
}