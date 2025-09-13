#pragma once

#include <chrono>

class BenchmarkTimer {
public:
    void start() {
        m_start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        m_end_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedTime() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(m_end_time - m_start_time);
        return duration.count() / 1000000.0; // Convert to seconds
    }
    
private:
    std::chrono::high_resolution_clock::time_point m_start_time;
    std::chrono::high_resolution_clock::time_point m_end_time;
};

struct BenchmarkResults {
    double total_time;
    double avg_time_per_step;
    double steps_per_second;
    double particle_steps_per_second;
    int num_particles;
    int num_steps;
};

BenchmarkResults runBenchmark(int num_particles, int num_steps, bool enable_visualization = false);