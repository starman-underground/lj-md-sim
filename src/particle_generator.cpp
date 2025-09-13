#include "particle_generator.hpp"
#include <cmath>
#include <iostream>

ParticleGenerator::ParticleGenerator(unsigned int seed)
    : m_rng(seed), m_uniform_dist(-1.0f, 1.0f), m_normal_dist(0.0f, 1.0f) {
}

ParticleConfiguration ParticleGenerator::generateRandomConfiguration(
    int num_particles, float temperature, float box_size, float mass) {
    
    ParticleConfiguration config;
    config.positions.reserve(num_particles);
    config.velocities.reserve(num_particles);
    
    float half_box = box_size * 0.5f;
    float inv_mass = 1.0f / mass;
    
    for (int i = 0; i < num_particles; ++i) {
        // Random position within box
        float4 pos;
        pos.x = m_uniform_dist(m_rng) * half_box;
        pos.y = m_uniform_dist(m_rng) * half_box;
        pos.z = m_uniform_dist(m_rng) * half_box;
        pos.w = mass;
        
        // Maxwell-Boltzmann velocity distribution
        float3 vel_3d = generateMBVelocity(temperature, mass);
        float4 vel;
        vel.x = vel_3d.x;
        vel.y = vel_3d.y;
        vel.z = vel_3d.z;
        vel.w = inv_mass;
        
        config.positions.push_back(pos);
        config.velocities.push_back(vel);
    }
    
    // Remove center of mass motion
    removeCOMMotion(config.velocities, config.positions);
    
    std::cout << "Generated " << num_particles << " particles in random configuration\n";
    std::cout << "Box size: " << box_size << ", Temperature: " << temperature << "\n";
    
    return config;
}

ParticleConfiguration ParticleGenerator::generateLatticeConfiguration(
    int particles_per_side, float lattice_spacing, float temperature, float mass) {
    
    int num_particles = particles_per_side * particles_per_side * particles_per_side;
    
    ParticleConfiguration config;
    config.positions.reserve(num_particles);
    config.velocities.reserve(num_particles);
    
    float inv_mass = 1.0f / mass;
    float box_size = particles_per_side * lattice_spacing;
    float offset = -box_size * 0.5f + lattice_spacing * 0.5f;
    
    for (int i = 0; i < particles_per_side; ++i) {
        for (int j = 0; j < particles_per_side; ++j) {
            for (int k = 0; k < particles_per_side; ++k) {
                // Lattice position
                float4 pos;
                pos.x = offset + i * lattice_spacing;
                pos.y = offset + j * lattice_spacing;
                pos.z = offset + k * lattice_spacing;
                pos.w = mass;
                
                // Maxwell-Boltzmann velocity distribution
                float3 vel_3d = generateMBVelocity(temperature, mass);
                float4 vel;
                vel.x = vel_3d.x;
                vel.y = vel_3d.y;
                vel.z = vel_3d.z;
                vel.w = inv_mass;
                
                config.positions.push_back(pos);
                config.velocities.push_back(vel);
            }
        }
    }
    
    // Remove center of mass motion
    removeCOMMotion(config.velocities, config.positions);
    
    std::cout << "Generated " << num_particles << " particles in " 
              << particles_per_side << "³ lattice configuration\n";
    std::cout << "Lattice spacing: " << lattice_spacing << ", Temperature: " << temperature << "\n";
    
    return config;
}

ParticleConfiguration ParticleGenerator::generateSphericalShell(
    int num_particles, float inner_radius, float outer_radius, float temperature, float mass) {
    
    ParticleConfiguration config;
    config.positions.reserve(num_particles);
    config.velocities.reserve(num_particles);
    
    float inv_mass = 1.0f / mass;
    
    for (int i = 0; i < num_particles; ++i) {
        // Generate random point on unit sphere
        float theta = 2.0f * M_PI * m_uniform_dist(m_rng) * 0.5f + M_PI;  // [0, 2π]
        float phi = std::acos(2.0f * (m_uniform_dist(m_rng) * 0.5f + 0.5f) - 1.0f);  // [0, π]
        
        // Random radius between inner and outer
        float r_range = outer_radius - inner_radius;
        float r = inner_radius + r_range * (m_uniform_dist(m_rng) * 0.5f + 0.5f);
        
        // Convert to Cartesian coordinates
        float4 pos;
        pos.x = r * std::sin(phi) * std::cos(theta);
        pos.y = r * std::sin(phi) * std::sin(theta);
        pos.z = r * std::cos(phi);
        pos.w = mass;
        
        // Maxwell-Boltzmann velocity distribution
        float3 vel_3d = generateMBVelocity(temperature, mass);
        float4 vel;
        vel.x = vel_3d.x;
        vel.y = vel_3d.y;
        vel.z = vel_3d.z;
        vel.w = inv_mass;
        
        config.positions.push_back(pos);
        config.velocities.push_back(vel);
    }
    
    // Remove center of mass motion
    removeCOMMotion(config.velocities, config.positions);
    
    std::cout << "Generated " << num_particles << " particles in spherical shell\n";
    std::cout << "Inner radius: " << inner_radius << ", Outer radius: " << outer_radius 
              << ", Temperature: " << temperature << "\n";
    
    return config;
}

float3 ParticleGenerator::generateMBVelocity(float temperature, float mass) {
    // Maxwell-Boltzmann distribution: σ = sqrt(kT/m)
    // Using reduced units where k = 1
    float sigma = std::sqrt(temperature / mass);
    
    float3 velocity;
    velocity.x = m_normal_dist(m_rng) * sigma;
    velocity.y = m_normal_dist(m_rng) * sigma;
    velocity.z = m_normal_dist(m_rng) * sigma;
    
    return velocity;
}

void ParticleGenerator::removeCOMMotion(std::vector<float4>& velocities, 
                                      const std::vector<float4>& positions) {
    if (velocities.empty()) return;
    
    // Calculate center of mass velocity
    float3 com_velocity = {0.0f, 0.0f, 0.0f};
    float total_mass = 0.0f;
    
    for (size_t i = 0; i < velocities.size(); ++i) {
        float mass = positions[i].w;
        com_velocity.x += velocities[i].x * mass;
        com_velocity.y += velocities[i].y * mass;
        com_velocity.z += velocities[i].z * mass;
        total_mass += mass;
    }
    
    com_velocity.x /= total_mass;
    com_velocity.y /= total_mass;
    com_velocity.z /= total_mass;
    
    // Subtract COM velocity from all particles
    for (auto& vel : velocities) {
        vel.x -= com_velocity.x;
        vel.y -= com_velocity.y;
        vel.z -= com_velocity.z;
    }
    
    std::cout << "Removed center of mass motion: v_com = (" 
              << com_velocity.x << ", " << com_velocity.y << ", " << com_velocity.z << ")\n";
}