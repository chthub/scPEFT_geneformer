#!/bin/bash
#SBATCH --account=PCON0022 
#SBATCH --job-name=run_gene_cls
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/run_gene_cls_%j.out
#SBATCH --error=logs/run_gene_cls_%j.err    

# Parse command line arguments
DATASET_NAME=""
PROMPT_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dataset_name <dataset> --prompt_type <type>"
            echo "Available datasets: tf_dosage_sens_test, bivalent_promoters, tf_regulatory_range, N1_network"
            echo "Available prompt types: Gene_token_prompt, prefix_prompt, lora, encoder_prompt"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: --dataset_name is required"
    echo "Available datasets: tf_dosage_sens_test, bivalent_promoters, tf_regulatory_range, N1_network"
    exit 1
fi

if [[ -z "$PROMPT_TYPE" ]]; then
    echo "Error: --prompt_type is required"
    echo "Available prompt types: Gene_token_prompt, prefix_prompt, lora, encoder_prompt"
    exit 1
fi

cd /fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/example_py/

# Execute based on dataset name and prompt type
# Execute based on dataset name and prompt type
if [[ "$DATASET_NAME" == "tf_dosage_sens_test" ]]; then
    /fs/scratch/PCON0022/ch/geneformer_env/bin/python -u gene_cls2_prompt.py \
        --dataset_name "$DATASET_NAME" \
        --output_root ./outputs/tf_dosage_sens_test \
        --prompt_type "$PROMPT_TYPE"

elif [[ "$DATASET_NAME" == "bivalent_promoters" ]]; then
    /fs/scratch/PCON0022/ch/geneformer_env/bin/python -u gene_cls2_prompt.py \
        --dataset_name "$DATASET_NAME" \
        --output_root ./outputs/bivalent_promoters \
        --prompt_type "$PROMPT_TYPE"

elif [[ "$DATASET_NAME" == "tf_regulatory_range" ]]; then
    /fs/scratch/PCON0022/ch/geneformer_env/bin/python -u gene_cls2_prompt.py \
        --dataset_name "$DATASET_NAME" \
        --output_root ./outputs/tf_regulatory_range \
        --prompt_type "$PROMPT_TYPE" \
        --batch_size 8

elif [[ "$DATASET_NAME" == "N1_network" ]]; then
    /fs/scratch/PCON0022/ch/geneformer_env/bin/python -u gene_cls2_prompt.py \
        --dataset_name "$DATASET_NAME" \
        --output_root ./outputs/N1_network \
        --prompt_type "$PROMPT_TYPE" \
        --batch_size 8

else
    echo "Error: Unknown dataset_name '$DATASET_NAME'"
    echo "Available datasets: tf_dosage_sens_test, bivalent_promoters, tf_regulatory_range, N1_network"
    exit 1
fi

echo "Completed execution for dataset: $DATASET_NAME with prompt type: $PROMPT_TYPE"