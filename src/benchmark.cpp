#include "benchmark.hpp"
#include "lj_sim.cuh"
#include "particle_generator.hpp"
#include <iostream>

BenchmarkResults runBenchmark(int num_particles, int num_steps, bool enable_visualization) {
    std::cout << "Running benchmark with " << num_particles << " particles for " << num_steps << " steps\n";
    std::cout << "Visualization: " << (enable_visualization ? "ON" : "OFF") << "\n";
    
    // Initialize system
    LJSystem lj_system(num_particles, 1.0f, 1.0f, 2.5f);
    
    // Generate initial configuration
    ParticleGenerator generator;
    auto config = generator.generateRandomConfiguration(num_particles, 1.0f, 30.0f);
    
    // Set initial state
    lj_system.setPositions(config.positions.data());
    lj_system.setVelocities(config.velocities.data());
    
    // Benchmark parameters
    float dt = 0.001f;
    
    // Run benchmark
    BenchmarkTimer timer;
    timer.start();
    
    for (int step = 0; step < num_steps; ++step) {
        lj_system.step(dt);
        
        // Print progress every 10% of steps
        if (step % (num_steps / 10) == 0 && step > 0) {
            std::cout << "Progress: " << (100 * step / num_steps) << "%\n";
        }
    }
    
    timer.stop();
    
    // Calculate results
    BenchmarkResults results;
    results.total_time = timer.getElapsedTime();
    results.avg_time_per_step = results.total_time / num_steps;
    results.steps_per_second = num_steps / results.total_time;
    results.particle_steps_per_second = (static_cast<double>(num_particles) * num_steps) / results.total_time;
    results.num_particles = num_particles;
    results.num_steps = num_steps;
    
    // Print results
    std::cout << "\nBenchmark Results:\n";
    std::cout << "=================\n";
    std::cout << "Total time: " << results.total_time << " seconds\n";
    std::cout << "Average time per step: " << results.avg_time_per_step * 1000 << " ms\n";
    std::cout << "Steps per second: " << results.steps_per_second << "\n";
    std::cout << "Particle-steps per second: " << results.particle_steps_per_second << "\n";
    
    return results;
}

int main(int argc, char* argv[]) {
    // Default benchmark parameters
    int num_particles = 10000;
    int num_steps = 1000;
    bool enable_visualization = false;
    
    // Parse command line arguments
    if (argc > 1) num_particles = std::stoi(argv[1]);
    if (argc > 2) num_steps = std::stoi(argv[2]);
    if (argc > 3) enable_visualization = std::stoi(argv[3]);
    
    std::cout << "LJ Simulation Benchmark\n";
    std::cout << "======================\n";
    std::cout << "Particles: " << num_particles << "\n";
    std::cout << "Steps: " << num_steps << "\n";
    std::cout << "Visualization: " << (enable_visualization ? "ON" : "OFF") << "\n\n";
    
    try {
        // Run the benchmark
        auto results = runBenchmark(num_particles, num_steps, enable_visualization);
        
        std::cout << "\nBenchmark completed successfully!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}