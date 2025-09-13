#include <iostream>
#include <memory>
#include <chrono>
#include "mimir/mimir.hpp"
#include "lj_sim.cuh"
#include "particle_generator.hpp"
#include "benchmark.hpp"

using namespace mimir;

struct SimulationParams {
    float sigma = 1.0f;           // LJ characteristic length
    float epsilon = 1.0f;         // LJ energy well depth
    float cutoff = 2.5f;          // Cutoff distance (in units of sigma)
    float dt = 0.001f;            // Time step
    float temperature = 1.0f;     // Initial temperature
    int num_particles = 10000;    // Number of particles
    int max_steps = 10000;        // Maximum simulation steps
    bool enable_visualization = true;
    bool enable_benchmarking = false;
};

int main(int argc, char* argv[]) {
    SimulationParams params;
    
    // Parse command line arguments
    if (argc > 1) params.num_particles = std::stoi(argv[1]);
    if (argc > 2) params.max_steps = std::stoi(argv[2]);
    if (argc > 3) params.enable_visualization = std::stoi(argv[3]);
    if (argc > 4) params.enable_benchmarking = std::stoi(argv[4]);
    
    std::cout << "Lennard-Jones Particle Simulator\n";
    std::cout << "================================\n";
    std::cout << "Particles: " << params.num_particles << "\n";
    std::cout << "Max Steps: " << params.max_steps << "\n";
    std::cout << "Visualization: " << (params.enable_visualization ? "ON" : "OFF") << "\n";
    std::cout << "Benchmarking: " << (params.enable_benchmarking ? "ON" : "OFF") << "\n\n";
    
    try {
        // Initialize CUDA
        checkCuda(cudaSetDevice(0));
        
        // Setup Mimir if visualization is enabled
        InstanceHandle instance = nullptr;
        if (params.enable_visualization) {
            ViewerOptions options{};
            options.window.size = {1920, 1080};
            options.background_color = {0.1f, 0.1f, 0.2f, 1.0f};
            options.present.mode = PresentMode::Immediate;
            options.present.enable_sync = true;
            options.present.target_fps = 60;
            
            createInstance(options, &instance);
            setCameraPosition(instance, {0.0f, 0.0f, -50.0f});
        }
        
        // Initialize LJ simulation system
        LJSystem lj_system(params.num_particles, params.sigma, params.epsilon, params.cutoff);
        
        // Generate initial particle configuration
        ParticleGenerator generator;
        auto initial_config = generator.generateRandomConfiguration(
            params.num_particles, 
            params.temperature,
            30.0f  // box size
        );
        
        // Copy initial data to GPU
        lj_system.setPositions(initial_config.positions.data());
        lj_system.setVelocities(initial_config.velocities.data());
        
        // Setup Mimir visualization
        AllocHandle pos_alloc;
        ViewHandle particle_view;
        
        if (params.enable_visualization) {
            // Use Mimir's allocation for position data
            allocLinear(instance, (void**)&lj_system.d_positions, 
                       params.num_particles * sizeof(float4), &pos_alloc);
            
            // Create particle view
            ViewDescription view_desc {
                .type = ViewType::Markers,
                .domain = DomainType::Domain3D,
                .attributes = {
                    { AttributeType::Position, {
                        .source = pos_alloc,
                        .size = params.num_particles,
                        .format = FormatDescription::make<float4>(),
                    }}
                },
                .layout = Layout::make(params.num_particles),
                .visible = true,
                .default_color = {1.0f, 0.8f, 0.2f, 1.0f},
                .default_size = 0.5f,
            };
            
            createView(instance, &view_desc, &particle_view);
        }
        
        // Run simulation
        BenchmarkTimer timer;
        if (params.enable_benchmarking) {
            timer.start();
        }
        
        std::cout << "Starting simulation...\n";
        
        if (params.enable_visualization) {
            // Asynchronous rendering mode
            displayAsync(instance);
            
            for (int step = 0; step < params.max_steps && isRunning(instance); ++step) {
                prepareViews(instance);
                
                // Run LJ simulation step
                lj_system.step(params.dt);
                
                updateViews(instance);
                
                // Print progress every 1000 steps
                if (step % 1000 == 0) {
                    float ke = lj_system.getKineticEnergy();
                    float pe = lj_system.getPotentialEnergy();
                    std::cout << "Step " << step << ": KE=" << ke 
                             << ", PE=" << pe << ", Total=" << (ke + pe) << "\n";
                }
            }
        } else {
            // Headless mode
            for (int step = 0; step < params.max_steps; ++step) {
                lj_system.step(params.dt);
                
                if (step % 1000 == 0) {
                    float ke = lj_system.getKineticEnergy();
                    float pe = lj_system.getPotentialEnergy();
                    std::cout << "Step " << step << ": KE=" << ke 
                             << ", PE=" << pe << ", Total=" << (ke + pe) << "\n";
                }
            }
        }
        
        if (params.enable_benchmarking) {
            timer.stop();
            std::cout << "\nBenchmark Results:\n";
            std::cout << "Total time: " << timer.getElapsedTime() << " seconds\n";
            std::cout << "Steps per second: " << params.max_steps / timer.getElapsedTime() << "\n";
            std::cout << "Particle-steps per second: " 
                     << (params.num_particles * params.max_steps) / timer.getElapsedTime() << "\n";
        }
        
        // Cleanup
        if (params.enable_visualization) {
            exit(instance);
            destroyInstance(instance);
        }
        
        std::cout << "Simulation completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}