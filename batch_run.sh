#!/bin/bash

# Batch script for running LJ simulator benchmarks
# Usage: ./batch_run.sh output_file.csv

if [ $# -eq 0 ]; then
    echo "Usage: $0 <output_file.csv>"
    exit 1
fi

output_file=$1

# Simulation parameters
particle_counts=(1000 5000 10000 25000 50000)
step_counts=(1000 5000 10000)
visualization_modes=(0 1)  # 0 = off, 1 = on

# CSV header
echo "num_particles,num_steps,visualization,total_time,avg_time_per_step,steps_per_second,particle_steps_per_second" > $output_file

echo "Starting batch benchmark run..."
echo "Output file: $output_file"

# Run benchmarks
for particles in ${particle_counts[@]}; do
    for steps in ${step_counts[@]}; do
        for viz in ${visualization_modes[@]}; do
            echo "Running: particles=$particles, steps=$steps, visualization=$viz"
            
            # Run the benchmark and capture output
            ./build/bin/lj_benchmark $particles $steps $viz 0 1 >> $output_file
            
            if [ $? -ne 0 ]; then
                echo "Error running benchmark with particles=$particles, steps=$steps, viz=$viz"
            else
                echo "Completed successfully"
            fi
        done
    done
done

echo "Batch benchmark completed. Results saved to $output_file"