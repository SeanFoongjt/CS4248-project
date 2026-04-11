#!/bin/bash

MODEL_TYPE=""
POS_ARGS=""
POS_SUFFIX=""

# Parse command-line arguments flexibly
while [[ "$#" -gt 0 ]]; do
    case $1 in
        distilbert|roberta)
            MODEL_TYPE="$1"
            shift
            ;;
        --pos)
            POS_SUFFIX="_pos"
            shift # Consume the '--pos' flag itself
            
            # Collect all subsequent arguments until we hit another flag (starting with '-') 
            # or one of our recognised model types.
            while [[ "$#" -gt 0 && ! "$1" =~ ^- && "$1" != "distilbert" && "$1" != "roberta" ]]; do
                if [[ -z "$POS_ARGS" ]]; then
                    POS_ARGS="$1"
                else
                    POS_ARGS="$POS_ARGS $1"
                fi
                POS_SUFFIX="${POS_SUFFIX}_$1"
                shift
            done
            ;;
        *)
            echo "Error: Unrecognised parameter passed: $1"
            echo "Usage: ./submit_experiments.sh [roberta|distilbert] [--pos arg1 arg2 ...]"
            exit 1
            ;;
    esac
done

# Validate that a model type was provided
if [[ -z "$MODEL_TYPE" ]]; then
    echo "Error: Invalid or missing model type."
    echo "Usage: ./submit_experiments.sh [roberta|distilbert] [--pos arg1 arg2 ...]"
    exit 1
fi

# Format the POS string for export. If POS_ARGS has content, prepend the flag.
POS_EXPORT=""
if [[ -n "$POS_ARGS" ]]; then
    POS_EXPORT="--pos ${POS_ARGS}"
fi

# Define the variations
FORMATS=("headline" "headline_section" "all")
CONCEPTNET_OPTS=("with_conceptnet" "without_conceptnet")

# Define the absolute path to your working directory
SCRIPT_DIR="${HOME}/cs4248"

# Ensure logs directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Ensure logs directory exists
mkdir -p logs

for fmt in "${FORMATS[@]}"; do
    for cnet in "${CONCEPTNET_OPTS[@]}"; do
        # Define a unique run name for the output directory and logs
        RUN_NAME="${MODEL_TYPE}_${fmt}_${cnet}${POS_SUFFIX}"

        # Set the ConceptNet flag (empty string if we want to use ConceptNet)
        CNET_FLAG=""
        if [[ "$cnet" == "without_conceptnet" ]]; then
            CNET_FLAG="--no-conceptnet"
        fi

        mkdir -p "${SCRIPT_DIR}/outputs/${RUN_NAME}"
        echo "Submitting job for: ${RUN_NAME}"

        # Submit the job, overriding the job name and log files dynamically
        # The --export flag passes our loop variables directly into the Slurm script
        sbatch \
            --job-name="cs4248_${RUN_NAME}" \
            --output="${SCRIPT_DIR}/logs/${RUN_NAME}_%j.out" \
            --error="${SCRIPT_DIR}/logs/${RUN_NAME}_%j.err" \
            --export=ALL,FORMAT="${fmt}",CNET_FLAG="${CNET_FLAG}",RUN_NAME="${RUN_NAME}",MODEL_TYPE="${MODEL_TYPE}",POS_FLAG="${POS_EXPORT}" \
            "${SCRIPT_DIR}/slurm_general/sarcasm_tune.slurm"

    done
done

echo "All 6 jobs submitted successfully."
