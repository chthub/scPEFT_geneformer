#!/bin/bash
#SBATCH --account=PCON0022 
#SBATCH --job-name=run_gene_cls
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=logs/run_gene_cls_%j.out
#SBATCH --error=logs/run_gene_cls_%j.err    

# Function to display usage information
usage() {
    echo "Usage: $0 <data_name>"
    echo "Available data names:"
    echo "  tf_dosage_sens_test   - TF dosage sensitivity test"
    echo "  bivalent_promoters    - Bivalent promoters classification"
    echo "  tf_regulatory_range   - TF regulatory range classification"
    echo "  N1_network           - Notch1 network classification"
    echo "  multitask            - Run multitask classification"
    exit 1
}

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No data name provided."
    usage
fi

# Get the data name from command line argument
data_name="$1"

# Base paths
BASE_DIR="/fs/scratch/PCON0022/ch/Geneformer"
PYTHON_CMD="${BASE_DIR}/.venv/bin/python -u"
GENE_CLS_SCRIPT="${BASE_DIR}/examples/gene_cls1.py"
MULTITASK_SCRIPT="${BASE_DIR}/examples/multitask.py"

echo "Running gene classification for dataset: $data_name"

# Execute different commands based on data_name
case "$data_name" in
    "tf_dosage_sens_test")
        echo "Running TF dosage sensitivity test..."
        $PYTHON_CMD $GENE_CLS_SCRIPT
        ;;
    
    "bivalent_promoters")
        echo "Running bivalent promoters classification..."
        $PYTHON_CMD $GENE_CLS_SCRIPT \
            --input_data_file "${BASE_DIR}/examples/gene_inputs/example_input_files/gene_classification/bivalent_promoters/panglao_SRA553822-SRS2119548.dataset" \
            --output_prefix bivalent_promoters \
            --gene_dict_path "${BASE_DIR}/examples/gene_inputs/example_input_files/gene_classification/bivalent_promoters/bivalent_vs_lys4_only_genomewide.pickle"
        ;;
    
    "tf_regulatory_range")
        echo "Running TF regulatory range classification..."
        $PYTHON_CMD $GENE_CLS_SCRIPT \
            --input_data_file "${BASE_DIR}/examples/gene_inputs/example_input_files/gene_classification/tf_regulatory_range/iCM_diff_dropseq.dataset" \
            --output_prefix tf_regulatory_range \
            --gene_dict_path "${BASE_DIR}/examples/gene_inputs/example_input_files/gene_classification/tf_regulatory_range/tf_regulatory_range.pickle"
        ;;
    
    "N1_network")
        echo "Running Notch1 network classification..."
        $PYTHON_CMD $GENE_CLS_SCRIPT \
            --input_data_file "${BASE_DIR}/examples/gene_inputs/example_input_files/gene_classification/notch1_network/heart_atlas_endothelial_cells.dataset" \
            --output_prefix N1_network \
            --gene_dict_path "${BASE_DIR}/examples/gene_inputs/example_input_files/gene_classification/notch1_network/n1_network.pickle"
        ;;
    
    "multitask")
        echo "Running multitask classification..."
        $PYTHON_CMD $MULTITASK_SCRIPT
        ;;
    
    *)
        echo "Error: Unknown data name '$data_name'"
        usage
        ;;
esac

echo "Gene classification completed for dataset: $data_name"