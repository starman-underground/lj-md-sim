#include "lj_sim.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <iostream>

// Device constants
__constant__ LJParams d_lj_params;

// CUDA device functions
__device__ __forceinline__ float3 lj_force(float3 r_ij, float r_sq, const LJParams& params) {
    if (r_sq >= params.cutoff_sq || r_sq < 1e-6f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float sigma_sq = params.sigma * params.sigma;
    float sigma2_over_r2 = sigma_sq / r_sq;
    float sigma6_over_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
    float sigma12_over_r12 = sigma6_over_r6 * sigma6_over_r6;
    
    // Force magnitude: 24 * epsilon * (2 * sigma^12/r^12 - sigma^6/r^6) / r^2
    float force_mag = 24.0f * params.epsilon * (2.0f * sigma12_over_r12 - sigma6_over_r6) / r_sq;
    
    return make_float3(force_mag * r_ij.x, force_mag * r_ij.y, force_mag * r_ij.z);
}

__device__ __forceinline__ float lj_potential(float r_sq, const LJParams& params) {
    if (r_sq >= params.cutoff_sq) {
        return 0.0f;
    }
    
    float sigma_sq = params.sigma * params.sigma;
    float sigma2_over_r2 = sigma_sq / r_sq;
    float sigma6_over_r6 = sigma2_over_r2 * sigma2_over_r2 * sigma2_over_r2;
    float sigma12_over_r12 = sigma6_over_r6 * sigma6_over_r6;
    
    // Potential: 4 * epsilon * (sigma^12/r^12 - sigma^6/r^6)
    return 4.0f * params.epsilon * (sigma12_over_r12 - sigma6_over_r6);
}

// CUDA kernels
__global__ void lj_force_kernel(float4* positions, float3* forces, int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos_i = positions[i];
    float3 force_i = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int j = 0; j < num_particles; j++) {
        if (i != j) {
            float4 pos_j = positions[j];
            float3 r_ij = make_float3(
                pos_j.x - pos_i.x,
                pos_j.y - pos_i.y,
                pos_j.z - pos_i.z
            );
            
            float r_sq = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
            float3 f_ij = lj_force(r_ij, r_sq, d_lj_params);
            
            force_i.x += f_ij.x;
            force_i.y += f_ij.y;
            force_i.z += f_ij.z;
        }
    }
    
    forces[i] = force_i;
}

__global__ void integration_kernel(float4* positions, float4* velocities, float3* forces, 
                                  int num_particles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos = positions[i];
    float4 vel = velocities[i];
    float3 force = forces[i];
    
    float inv_mass = vel.w; // Stored in w component
    
    // Velocity Verlet integration
    // v(t + dt/2) = v(t) + a(t) * dt/2
    vel.x += force.x * inv_mass * dt * 0.5f;
    vel.y += force.y * inv_mass * dt * 0.5f;
    vel.z += force.z * inv_mass * dt * 0.5f;
    
    // r(t + dt) = r(t) + v(t + dt/2) * dt
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    
    positions[i] = pos;
    velocities[i] = vel;
}

__global__ void kinetic_energy_kernel(float4* velocities, int num_particles, float* ke_partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    
    float ke = 0.0f;
    if (i < num_particles) {
        float4 vel = velocities[i];
        float mass = 1.0f / vel.w; // mass = 1 / inv_mass
        ke = 0.5f * mass * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    }
    
    sdata[tid] = ke;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        ke_partial[blockIdx.x] = sdata[0];
    }
}

__global__ void potential_energy_kernel(float4* positions, int num_particles, float* pe_partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    
    float pe = 0.0f;
    if (i < num_particles) {
        float4 pos_i = positions[i];
        
        for (int j = i + 1; j < num_particles; j++) {
            float4 pos_j = positions[j];
            float r_sq = (pos_j.x - pos_i.x) * (pos_j.x - pos_i.x) +
                        (pos_j.y - pos_i.y) * (pos_j.y - pos_i.y) +
                        (pos_j.z - pos_i.z) * (pos_j.z - pos_i.z);
            
            pe += lj_potential(r_sq, d_lj_params);
        }
    }
    
    sdata[tid] = pe;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        pe_partial[blockIdx.x] = sdata[0];
    }
}

// Host class implementation
LJSystem::LJSystem(int num_particles, float sigma, float epsilon, float cutoff)
    : m_num_particles(num_particles), m_block_size(256) {
    
    m_grid_size = (num_particles + m_block_size - 1) / m_block_size;
    
    setLJParameters(sigma, epsilon, cutoff);
    allocateDeviceMemory();
}

LJSystem::~LJSystem() {
    freeDeviceMemory();
}

void LJSystem::allocateDeviceMemory() {
    size_t pos_size = m_num_particles * sizeof(float4);
    size_t vel_size = m_num_particles * sizeof(float4);
    size_t force_size = m_num_particles * sizeof(float3);
    
    checkCuda(cudaMalloc(&d_positions, pos_size));
    checkCuda(cudaMalloc(&d_velocities, vel_size));
    checkCuda(cudaMalloc(&d_forces, force_size));
}

void LJSystem::freeDeviceMemory() {
    if (d_positions) cudaFree(d_positions);
    if (d_velocities) cudaFree(d_velocities);
    if (d_forces) cudaFree(d_forces);
}

void LJSystem::setLJParameters(float sigma, float epsilon, float cutoff) {
    m_lj_params.sigma = sigma;
    m_lj_params.epsilon = epsilon;
    m_lj_params.cutoff = cutoff;
    m_lj_params.cutoff_sq = cutoff * cutoff;
    
    checkCuda(cudaMemcpyToSymbol(d_lj_params, &m_lj_params, sizeof(LJParams)));
}

void LJSystem::setPositions(const float4* host_positions) {
    checkCuda(cudaMemcpy(d_positions, host_positions, m_num_particles * sizeof(float4), cudaMemcpyHostToDevice));
}

void LJSystem::setVelocities(const float4* host_velocities) {
    checkCuda(cudaMemcpy(d_velocities, host_velocities, m_num_particles * sizeof(float4), cudaMemcpyHostToDevice));
}

void LJSystem::step(float dt) {
    computeForces();
    integrateMotion(dt);
}

void LJSystem::computeForces() {
    lj_force_kernel<<<m_grid_size, m_block_size>>>(d_positions, d_forces, m_num_particles);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

void LJSystem::integrateMotion(float dt) {
    integration_kernel<<<m_grid_size, m_block_size>>>(d_positions, d_velocities, d_forces, m_num_particles, dt);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

float LJSystem::getKineticEnergy() {
    float* d_ke_partial;
    checkCuda(cudaMalloc(&d_ke_partial, m_grid_size * sizeof(float)));
    
    kinetic_energy_kernel<<<m_grid_size, m_block_size>>>(d_velocities, m_num_particles, d_ke_partial);
    checkCuda(cudaGetLastError());
    
    // Final reduction on CPU (simple approach)
    std::vector<float> h_ke_partial(m_grid_size);
    checkCuda(cudaMemcpy(h_ke_partial.data(), d_ke_partial, m_grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_ke = 0.0f;
    for (float ke : h_ke_partial) {
        total_ke += ke;
    }
    
    checkCuda(cudaFree(d_ke_partial));
    return total_ke;
}

float LJSystem::getPotentialEnergy() {
    float* d_pe_partial;
    checkCuda(cudaMalloc(&d_pe_partial, m_grid_size * sizeof(float)));
    
    potential_energy_kernel<<<m_grid_size, m_block_size>>>(d_positions, m_num_particles, d_pe_partial);
    checkCuda(cudaGetLastError());
    
    // Final reduction on CPU (simple approach)
    std::vector<float> h_pe_partial(m_grid_size);
    checkCuda(cudaMemcpy(h_pe_partial.data(), d_pe_partial, m_grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_pe = 0.0f;
    for (float pe : h_pe_partial) {
        total_pe += pe;
    }
    
    checkCuda(cudaFree(d_pe_partial));
    return total_pe;
}

void LJSystem::getPositions(float4* host_positions) {
    checkCuda(cudaMemcpy(host_positions, d_positions, m_num_particles * sizeof(float4), cudaMemcpyDeviceToHost));
}

void LJSystem::getVelocities(float4* host_velocities) {
    checkCuda(cudaMemcpy(host_velocities, d_velocities, m_num_particles * sizeof(float4), cudaMemcpyDeviceToHost));
}