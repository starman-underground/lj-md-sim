#pragma once

#include <vector>
#include <random>
#include <cuda_runtime.h>

struct ParticleConfiguration {
    std::vector<float4> positions;  // x, y, z, mass
    std::vector<float4> velocities; // vx, vy, vz, inv_mass
};

class ParticleGenerator {
public:
    ParticleGenerator(unsigned int seed = 12345);
    
    // Generate random particle configuration
    ParticleConfiguration generateRandomConfiguration(
        int num_particles,
        float temperature = 1.0f,
        float box_size = 30.0f,
        float mass = 1.0f
    );
    
    // Generate particles in a crystalline lattice
    ParticleConfiguration generateLatticeConfiguration(
        int particles_per_side,
        float lattice_spacing = 1.5f,
        float temperature = 1.0f,
        float mass = 1.0f
    );
    
    // Generate particles in a spherical shell
    ParticleConfiguration generateSphericalShell(
        int num_particles,
        float inner_radius = 10.0f,
        float outer_radius = 15.0f,
        float temperature = 1.0f,
        float mass = 1.0f
    );
    
private:
    std::mt19937 m_rng;
    std::uniform_real_distribution<float> m_uniform_dist;
    std::normal_distribution<float> m_normal_dist;
    
    // Generate Maxwell-Boltzmann velocity distribution
    float3 generateMBVelocity(float temperature, float mass);
    
    // Remove center of mass motion
    void removeCOMMotion(std::vector<float4>& velocities, 
                        const std::vector<float4>& positions);
};