#!/bin/bash

# Script to run simulations on all JSON files in data2 folder
# Runs 5 simulations in parallel, processing all files in batches

# Configuration
DATA_DIR="data2"
OUTPUT_DIR="simulations"
MAX_PARALLEL=5
MAX_ROUNDS=10

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to run a single simulation
run_simulation() {
    local file="$1"
    # This line extracts the base name of the file (removing the directory path and the .json extension)
    # For example, if file="data2/example_scenario.json", then basename="example_scenario"
    local basename=$(basename "$file" .json)
    local output_file="$OUTPUT_DIR/sim_${basename}_gemini-2.5-pro.json"
    
    echo "Starting simulation for: $file"
    python simulate_agents.py \
        --scenario_file "$file" \
        --llm gemini \
        --max-rounds $MAX_ROUNDS \
        --output "$output_file" \
        > "logs_${basename}.txt" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed: $file"
    else
        echo "❌ Failed: $file (exit code: $exit_code)"
    fi
}

# Get all JSON files from data2 directory
files=($(find "$DATA_DIR" -name "*.json" | sort))

# Count total files
total_files=${#files[@]}
echo "Found $total_files JSON files to process"
echo "Running $MAX_PARALLEL simulations in parallel"
echo "Total batches: $(( (total_files + MAX_PARALLEL - 1) / MAX_PARALLEL ))"
echo "=========================================="

# Process files in batches
for ((i=0; i<total_files; i+=MAX_PARALLEL)); do
    batch_num=$((i/MAX_PARALLEL + 1))
    echo "Processing batch $batch_num (files $((i+1))-$((i+MAX_PARALLEL > total_files ? total_files : i+MAX_PARALLEL)))"
    
    # Start up to MAX_PARALLEL background jobs
    for ((j=0; j<MAX_PARALLEL && i+j<total_files; j++)); do
        run_simulation "${files[i+j]}" &
    done
    
    # Wait for all background jobs in this batch to complete
    wait
    
    echo "Batch $batch_num completed"
    echo "------------------------------------------"
done

echo "=========================================="
echo "All simulations completed!"
echo "Check the $OUTPUT_DIR directory for results"
echo "Check individual log files (logs_*.txt) for detailed output"