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

cd /fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/example_py/
/fs/scratch/PCON0022/ch/geneformer_env/bin/python gene_cls2_prompt.py --prompt_type encoder_prompt


# Gene_token_prompt
# prefix_prompt
# lora
# encoder_prompt