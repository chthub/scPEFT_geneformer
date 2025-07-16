
# sbatch run_gene_ls_prompt.sh --dataset_name tf_dosage_sens_test --prompt_type encoder_prompt
# sbatch run_gene_ls_prompt.sh --dataset_name tf_dosage_sens_test --prompt_type lora
# sbatch run_gene_ls_prompt.sh --dataset_name tf_dosage_sens_test --prompt_type prefix_prompt
# sbatch run_gene_ls_prompt.sh --dataset_name tf_dosage_sens_test --prompt_type Gene_token_prompt



sbatch run_gene_ls_prompt.sh --dataset_name bivalent_promoters --prompt_type encoder_prompt
sbatch run_gene_ls_prompt.sh --dataset_name bivalent_promoters --prompt_type lora
sbatch run_gene_ls_prompt.sh --dataset_name bivalent_promoters --prompt_type prefix_prompt
sbatch run_gene_ls_prompt.sh --dataset_name bivalent_promoters --prompt_type Gene_token_prompt


sbatch run_gene_ls_prompt.sh --dataset_name tf_regulatory_range --prompt_type encoder_prompt
sbatch run_gene_ls_prompt.sh --dataset_name tf_regulatory_range --prompt_type lora
sbatch run_gene_ls_prompt.sh --dataset_name tf_regulatory_range --prompt_type prefix_prompt
sbatch run_gene_ls_prompt.sh --dataset_name tf_regulatory_range --prompt_type Gene_token_prompt

sbatch run_gene_ls_prompt.sh --dataset_name N1_network --prompt_type encoder_prompt
sbatch run_gene_ls_prompt.sh --dataset_name N1_network --prompt_type lora
sbatch run_gene_ls_prompt.sh --dataset_name N1_network --prompt_type prefix_prompt
sbatch run_gene_ls_prompt.sh --dataset_name N1_network --prompt_type Gene_token_prompt

