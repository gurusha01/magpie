#!/bin/bash

# Batch Analysis Script
# This script runs analysis on all simulations or specific ones

set -e  # Exit on error

# Default values
OUTPUT_DIR="analysis"
LLM_TYPE="gemini"
MODEL_NAME="gemini-2.0-flash-exp"

# Function to print usage
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --all                   Analyze all simulations in simulations/ directory
    --scenario NAME         Analyze specific scenario (e.g., academic_job)
    --simulation FILE       Path to specific simulation file
    --guideline TYPE        Guideline type (implicit or explicit)
    --output-dir DIR        Output directory (default: analysis)
    --llm TYPE              Evaluator LLM (gemini|openai|together|bedrock, default: gemini)
    --model NAME            Model name (default: gemini-2.0-flash-exp)
    --help                  Show this help message

Examples:
    # Analyze all simulations
    $0 --all

    # Analyze all simulations with OpenAI
    $0 --all --llm openai --model gpt-4o

    # Use AWS Bedrock
    $0 --all --llm bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0

    # Analyze a specific simulation
    $0 --scenario academic_job \\
       --simulation simulations/explicit-data2/gemini/sim_academic_job.json \\
       --guideline explicit

    # Analyze all explicit guideline simulations only
    $0 --explicit-only

    # Analyze all implicit guideline simulations only
    $0 --implicit-only
EOF
}

# Parse arguments
ANALYZE_ALL=false
SCENARIO=""
SIMULATION=""
GUIDELINE=""
EXPLICIT_ONLY=false
IMPLICIT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ANALYZE_ALL=true
            shift
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --simulation)
            SIMULATION="$2"
            shift 2
            ;;
        --guideline)
            GUIDELINE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --llm)
            LLM_TYPE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --explicit-only)
            EXPLICIT_ONLY=true
            shift
            ;;
        --implicit-only)
            IMPLICIT_ONLY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multi-Agent Simulation Analysis"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Evaluator LLM: $LLM_TYPE ($MODEL_NAME)"
echo "=========================================="
echo ""

# Run analysis
if [ "$ANALYZE_ALL" = true ]; then
    echo "Analyzing all simulations..."
    python analysis.py --all --output-dir "$OUTPUT_DIR" --llm "$LLM_TYPE" --model "$MODEL_NAME"
elif [ "$EXPLICIT_ONLY" = true ]; then
    echo "Analyzing explicit guideline simulations only..."
    for llm_dir in simulations/explicit-data2/*/; do
        llm_name=$(basename "$llm_dir")
        echo "Processing $llm_name simulations..."
        for sim_file in "$llm_dir"*.json; do
            if [ -f "$sim_file" ]; then
                scenario_name=$(basename "$sim_file" .json | sed 's/^sim_//')
                echo "  - $scenario_name"
                python analysis.py \
                    --scenario "$scenario_name" \
                    --simulation "$sim_file" \
                    --guideline explicit \
                    --output-dir "$OUTPUT_DIR" \
                    --llm "$LLM_TYPE" \
                    --model "$MODEL_NAME" || echo "    Failed to analyze $sim_file"
            fi
        done
    done
elif [ "$IMPLICIT_ONLY" = true ]; then
    echo "Analyzing implicit guideline simulations only..."
    for llm_dir in simulations/implicit-data2/*/; do
        llm_name=$(basename "$llm_dir")
        echo "Processing $llm_name simulations..."
        for sim_file in "$llm_dir"*.json; do
            if [ -f "$sim_file" ]; then
                scenario_name=$(basename "$sim_file" .json | sed 's/^sim_//')
                echo "  - $scenario_name"
                python analysis.py \
                    --scenario "$scenario_name" \
                    --simulation "$sim_file" \
                    --guideline implicit \
                    --output-dir "$OUTPUT_DIR" \
                    --llm "$LLM_TYPE" \
                    --model "$MODEL_NAME" || echo "    Failed to analyze $sim_file"
            fi
        done
    done
elif [ -n "$SCENARIO" ] && [ -n "$SIMULATION" ] && [ -n "$GUIDELINE" ]; then
    echo "Analyzing single simulation: $SCENARIO"
    python analysis.py \
        --scenario "$SCENARIO" \
        --simulation "$SIMULATION" \
        --guideline "$GUIDELINE" \
        --output-dir "$OUTPUT_DIR" \
        --llm "$LLM_TYPE" \
        --model "$MODEL_NAME"
else
    echo "Error: Invalid arguments"
    echo ""
    print_usage
    exit 1
fi

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Summary of results:"
ls -lh "$OUTPUT_DIR" | tail -n +2 | wc -l | xargs echo "Total analysis files:"
