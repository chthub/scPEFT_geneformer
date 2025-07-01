import sys
import os


# cd /fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/example_py/
# /fs/scratch/PCON0022/ch/geneformer_env/bin/python gene_cls2_prompt.py


# Debug current paths
print("Current working directory:", os.getcwd())
print("Parent directory:", os.path.dirname(os.getcwd()))

# Add the transformerslocal directory to the Python path
transformers_path = os.path.join(os.path.dirname(os.getcwd()), 'transformerslocal')
print("Transformers path:", transformers_path)
print("Path exists:", os.path.exists(transformers_path))

if transformers_path not in sys.path:
    sys.path.insert(0, transformers_path)

print("Updated sys.path:")
for path in sys.path[:5]:  # Show first 5 paths
    print(f"  {path}")


parent_path = os.path.dirname(os.getcwd())
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
print(f"Added parent path: {parent_path}")

from transformerslocal.src.transformers.models.bert.modeling_bert import BertForTokenClassification
from transformerslocal.src.transformers import EarlyStoppingCallback, TrainerCallback

from transformers import Trainer
from transformers.training_args import TrainingArguments

# Import DataCollatorForGeneClassification from the original Geneformer directory
import sys
geneformer_path = "/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft"
if geneformer_path not in sys.path:
    sys.path.insert(0, geneformer_path)

# Import directly from the collator module to avoid name conflicts
from geneformer.collator_for_classification import DataCollatorForGeneClassification

import argparse, json, yaml, pickle, random, datetime, pathlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn
from datasets import load_from_disk

import loralib as lora
import pandas as pd
from sklearn.model_selection import KFold

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

print("Import successful with parent path!")

# -----------------------------------------------------------
# Helper Classes and Functions
# -----------------------------------------------------------

class LearningRateResetCallback(TrainerCallback):
    """Custom callback to reset learning rate if loss plateaus"""
    
    def __init__(self, patience=200, factor=0.5, min_lr=1e-6):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        current_loss = logs.get("train_loss", float('inf'))
        
        if current_loss < self.best_loss - 1e-4:  # Improvement threshold
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            # Reset learning rate
            current_lr = state.learning_rate if hasattr(state, 'learning_rate') else args.learning_rate
            new_lr = max(current_lr * self.factor, self.min_lr)
            
            if new_lr > self.min_lr and hasattr(kwargs.get('optimizer'), 'param_groups'):
                for param_group in kwargs['optimizer'].param_groups:
                    param_group['lr'] = new_lr
                print(f"🔄 Learning rate reset from {current_lr:.2e} to {new_lr:.2e} due to loss plateau")
                self.wait_count = 0
                self.best_loss = current_loss

def warm_up_model(model, data_collator, warmup_dataset, device="cuda"):
    """Perform a few forward passes to warm up the model and check gradients"""
    print("🔥 Warming up model with sample data...")
    
    model.train()
    from torch.utils.data import DataLoader
    
    # Create a small dataloader for warmup
    warmup_loader = DataLoader(warmup_dataset.select(range(min(10, len(warmup_dataset)))), 
                               batch_size=2, collate_fn=data_collator)
    
    # Perform a few forward passes
    for i, batch in enumerate(warmup_loader):
        if i >= 3:  # Only do 3 warmup batches
            break
            
        # Move to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Forward pass only
        with torch.no_grad():
            outputs = model(**{k: v for k, v in batch.items() if k in 
                             ['input_ids', 'attention_mask', 'labels']})
            print(f"  Warmup batch {i+1}: loss = {outputs.loss.item():.4f}")
    
    print("✅ Model warmup completed!")
    return model

def inspect_training_data(dataset, class_id_dict, id_class_dict, dataset_name=""):
    """Inspect the training data for debugging purposes"""
    print(f"\n🔍 INSPECTING {dataset_name} DATA")
    print("="*60)
    
    total_samples = len(dataset)
    total_tokens = 0
    total_labels = 0
    class_counts = {}
    
    for i, sample in enumerate(dataset):
        if i >= 1000:  # Sample first 1000 for speed
            break
            
        labels = sample["labels"]
        valid_labels = [l for l in labels if l != -100]
        
        total_tokens += len(labels)
        total_labels += len(valid_labels)
        
        for label in valid_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"Dataset size: {total_samples} samples")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total valid labels: {total_labels:,}")
    print(f"Average tokens per sample: {total_tokens / min(1000, total_samples):.1f}")
    print(f"Average labels per sample: {total_labels / min(1000, total_samples):.1f}")
    print(f"Label coverage: {100 * total_labels / total_tokens:.2f}%")
    
    print(f"\nClass distribution (top 10):")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (class_id, count) in enumerate(sorted_classes[:10]):
        class_name = id_class_dict.get(class_id, f"Unknown_{class_id}")
        percentage = 100 * count / total_labels
        print(f"  {i+1:2d}. Class {class_id} ({class_name}): {count:,} ({percentage:.2f}%)")
    
    # Check for class imbalance
    if len(sorted_classes) > 1:
        majority_count = sorted_classes[0][1]
        minority_count = sorted_classes[-1][1]
        imbalance_ratio = majority_count / minority_count
        print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 100:
            print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print("   This can cause poor learning. Consider:")
            print("   - Using focal loss")
            print("   - Increasing learning rate")
            print("   - Using class weights")
        elif imbalance_ratio > 10:
            print("⚠️  Moderate class imbalance detected")
    
    print("="*60)
    return class_counts

# -----------------------------------------------------------
# 1 ▸ CLI
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", default="/fs/scratch/PCON0022/ch/Geneformer/examples/example_input_files/gc-30M_sample50k.dataset")
parser.add_argument("--gene_class_dict", default="/fs/scratch/PCON0022/ch/Geneformer/examples/example_input_files/dosage_sensitivity_TFs.pickle")
parser.add_argument("--token_dict", 
                # default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/Geneformer/geneformer/token_dictionary_gc95M.pkl")
                default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl")
parser.add_argument("--ckpt_dir", 
                default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/Pretrain_ckpts/Pretrain_ckpts/geneformer-12L-30M-prompt")
                # default="/fs/scratch/PCON0022/ch/Geneformer/gf-6L-30M-i2048")
parser.add_argument("--output_root", default="/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/example_py/outputs")

parser.add_argument("--epochs", type=int, default=1)  # Increase epochs for better convergence
parser.add_argument("--batch_size", type=int, default=8)  # Increase batch size for more stable gradients
parser.add_argument("--lr",        type=float, default=5e-4)  # Higher learning rate for better convergence
parser.add_argument("--seed",      type=int, default=42)
parser.add_argument("--n_folds",   type=int, default=5)

parser.add_argument("--prompt_type", 
                    default="encoder_prompt")


# args = parser.parse_args('')
args = parser.parse_args()


torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

# Clear GPU cache before starting
torch.cuda.empty_cache()

# -----------------------------------------------------------
# 2 ▸ Load data & prepare for K-fold cross validation
# -----------------------------------------------------------
full_ds = load_from_disk(args.dataset_file).shuffle(seed=args.seed)  # one .dataset only

# -----------------------------------------------------------
# 3 ▸ Dict helpers
# -----------------------------------------------------------
def load_dict(pth):
    pth = pathlib.Path(pth)
    with open(pth, "rb" if pth.suffix == ".pkl" or pth.suffix == ".pickle" else "r") as f:
        return (
            pickle.load(f) if pth.suffix == ".pkl" or pth.suffix == ".pickle"
            else json.load(f) if pth.suffix == ".json"
            else yaml.safe_load(f)
        )

gene_class_dict = load_dict(args.gene_class_dict)      # {label: [ENS,…]}
token_dict      = load_dict(args.token_dict)           # {ENS: int_id}

# ↪ map gene token-id ➜ class-label
inverse_gene_dict = {
    token_dict[g]: cls for cls, genes in gene_class_dict.items() for g in genes if g in token_dict
}
class_id_dict = {cls: i for i, cls in enumerate(gene_class_dict.keys())}
id_class_dict = {v: k for k, v in class_id_dict.items()}

def label_example(ex):
    ex["labels"] = [
        class_id_dict.get(inverse_gene_dict.get(tok, None), -100)
        for tok in ex["input_ids"]
    ]
    return ex

# filter out cells without any labelled genes, then add "labels"
target_tokens = set(inverse_gene_dict.keys())
def keep_cell(ex): return not target_tokens.isdisjoint(ex["input_ids"])
filtered_ds = full_ds.filter(keep_cell, num_proc=16).map(label_example, num_proc=16)

print(f"Total samples after filtering: {len(filtered_ds)}")

# -----------------------------------------------------------
# 4 ▸ Collator 
# -----------------------------------------------------------
# Fix: Create the collator without token_dictionary parameter and patch it afterward
data_collator = DataCollatorForGeneClassification()

# Update the global token_dictionary in the collator modules to use our token_dict
import sys
for module_name in list(sys.modules.keys()):
    if 'geneformer' in module_name and hasattr(sys.modules[module_name], 'token_dictionary'):
        sys.modules[module_name].token_dictionary = token_dict

# Also update the tokenizer within the data_collator if it exists
if hasattr(data_collator.tokenizer, 'token_dictionary'):
    data_collator.tokenizer.token_dictionary = token_dict

# Update special token IDs in the precollator
data_collator.tokenizer.mask_token_id = token_dict.get("<mask>")
data_collator.tokenizer.pad_token_id = token_dict.get("<pad>")
data_collator.tokenizer.all_special_ids = [
    token_dict.get("<mask>"),
    token_dict.get("<pad>")
]

print("Data collator initialized and token dictionary updated.")

# -----------------------------------------------------------
# 5 ▸ Model creation function
# -----------------------------------------------------------
def create_model():
    """Create a fresh model for each fold with enhanced parameter configuration"""
    
    if args.prompt_type == 'encoder_prompt':
        config_path='/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/Pretrain_ckpts/Pretrain_ckpts/geneformer-12L-30M-prompt/config.json'
    elif args.prompt_type == 'lora':
        config_path='/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/Pretrain_ckpts/Pretrain_ckpts/config_lora.json'
    elif args.prompt_type == 'prefix_prompt':
        config_path='/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/Pretrain_ckpts/Pretrain_ckpts/config_prefix.json'
    elif args.prompt_type == 'Gene_token_prompt':
        config_path='/fs/scratch/PCON0022/ch/scPEFT_reproduction/geneformer_peft/Pretrain_ckpts/Pretrain_ckpts/config_token.json'
    else:
        raise ValueError(f"Unknown prompt type: {args.prompt_type}")
    
    model = BertForTokenClassification.from_pretrained(
        args.ckpt_dir,
        num_labels=len(class_id_dict),
        ignore_mismatched_sizes=False,
        config=config_path
    ).to("cuda")

    prompt_types = [p.strip() for p in model.config.prompt_type.split(",") if p.strip()]
    print(f"Loaded modelPrompt types: {prompt_types}")
    
    # Debug: Check ALL parameter names to understand the model structure
    print("Checking model structure...")
    all_param_names = []
    adapter_count = 0
    for name, param in model.named_parameters():
        all_param_names.append(name)
        if any(pattern in name for pattern in ["Space_Adapter", "MLP_Adapter", "adapter"]):
            adapter_count += 1
    
    print(f"Total parameters: {len(all_param_names)}")
    print(f"Adapter parameters found: {adapter_count}")
    
    # Show a few sample parameter names for verification
    print("Sample parameter names:")
    for name in all_param_names[:5]:
        print(f"  {name}")
    print("  ...")
    for name in all_param_names[-3:]:
        print(f"  {name}")
    
    # First, set all parameters to not require gradients
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    trainable_params = []
    
    if "lora" in prompt_types:
        print("Applying LoRA configuration...")
        # Apply LoRA configuration first
        lora.mark_only_lora_as_trainable(model, bias="lora_only")
        
        # Then fine-tune the configuration
        for name, param in model.named_parameters():
            if "lora_key" in name:
                param.requires_grad = False
            elif any(keyword in name for keyword in ["lora_A", "lora_B", "classifier"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (LoRA): {name}")

    if "Gene_token_prompt" in prompt_types:
        print("Applying Gene_token_prompt configuration...")
        for name, param in model.named_parameters():
            # Look for adapter patterns more broadly
            if any(pattern in name for pattern in ["adapter", "bert.adapter", "classifier"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (Gene_token): {name}")
                
    if "encoder_prompt" in prompt_types:
        print("Applying encoder_prompt configuration...")
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in ["Space_Adapter", "MLP_Adapter", "adapter", "classifier"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (encoder): {name}")
                
    if "prefix_prompt" in prompt_types:
        print("Applying prefix_prompt configuration...")
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in ["prompt_embeddings", "prompt", "classifier"]):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (prefix): {name}")
    
    # Enhanced fallback: Look for any adapter-like parameters and MORE aggressive trainable parameter selection
    if trainable_count == 0:
        print("Warning: No prompt-specific parameters found. Searching for adapter-like parameters...")
        adapter_patterns = ["adapter", "prompt", "lora", "classifier"]
        
        for name, param in model.named_parameters():
            if any(pattern in name.lower() for pattern in adapter_patterns):
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (fallback): {name}")
    
    # MORE AGGRESSIVE: Make additional layers trainable for better learning
    if trainable_count < 10000:  # If we have very few trainable parameters
        print("Very few trainable parameters found. Making additional layers trainable...")
        layer_patterns = ["layer.11", "layer.10", "pooler", "embeddings.layer_norm"]  # Last 2 layers + pooler + layer norm
        
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in layer_patterns):
                if not param.requires_grad:  # Only set if not already trainable
                    param.requires_grad = True
                    trainable_count += param.numel()
                    trainable_params.append(name)
                    print(f"  Set trainable (additional layer): {name}")
    
    # Final fallback: Make classifier trainable
    if trainable_count == 0:
        print("Final fallback: Making classifier layers trainable...")
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(name)
                print(f"  Set trainable (final fallback): {name}")
    
    # Final verification
    total_params = sum(p.numel() for p in model.parameters())
    actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Final parameter status:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {actual_trainable:,}")
    print(f"  Percentage trainable: {100 * actual_trainable / total_params:.2f}%")
    print(f"  Trainable parameter names: {trainable_params}")
    
    if actual_trainable == 0:
        raise ValueError("ERROR: No parameters are set to trainable! This will cause the gradient warning.")
    
    return model, prompt_types, trainable_params

def count_trainable(model, trainable_param_names):
    """Debug function to count and display trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable / total params: {train:,} / {total:,}")
    
    # List specific trainable parameters
    actual_trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            actual_trainable_params.append(name)
    
    if actual_trainable_params:
        print("Actually trainable parameters:")
        for name in actual_trainable_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(actual_trainable_params) > 10:
            print(f"  ... and {len(actual_trainable_params) - 10} more")
    else:
        print("WARNING: No trainable parameters found!")
    
    # Check consistency with expected trainable parameters
    expected_set = set(trainable_param_names)
    actual_set = set(actual_trainable_params)
    
    if expected_set != actual_set:
        print("WARNING: Mismatch between expected and actual trainable parameters!")
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if missing:
            print(f"  Missing from actual: {missing}")
        if extra:
            print(f"  Extra in actual: {extra}")
    else:
        print("✓ Trainable parameters match expectations")
    
    return train, total

# -----------------------------------------------------------
# 5.5 ▸ Parameter Update Test Function
# -----------------------------------------------------------
def test_parameter_updates(model, data_collator, train_dataset, trainable_param_names):
    """Test if parameters are actually being updated with a single forward/backward pass"""
    print("\n" + "="*60)
    print("TESTING PARAMETER UPDATES WITH SINGLE FORWARD/BACKWARD PASS")
    print("="*60)
    
    # Get a single batch from the dataset
    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
    batch = next(iter(dataloader))
    
    # Move batch to device
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to("cuda")
    
    # Store initial parameter values
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in trainable_param_names:
            initial_params[name] = param.data.clone()
            print(f"Tracking parameter: {name}")
    
    # Set up optimizer manually for this test
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad and n in trainable_param_names]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    
    print(f"\nOptimizer setup:")
    print(f"  Number of parameter groups: {len(optimizer.param_groups)}")
    print(f"  Total parameters in optimizer: {sum(len(group['params']) for group in optimizer.param_groups)}")
    
    # Clear gradients
    optimizer.zero_grad()
    
    # Forward pass
    print(f"\nPerforming forward pass...")
    
    # Filter batch to only include model-expected keys
    model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
                       'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
                       'output_hidden_states', 'return_dict'}
    
    filtered_batch = {k: v for k, v in batch.items() if k in model_input_keys}
    
    outputs = model(**filtered_batch)
    
    # Calculate loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(outputs.logits.view(-1, model.num_labels), filtered_batch["labels"].view(-1))
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    print(f"Performing backward pass...")
    loss.backward()
    
    # Check gradients
    print(f"\nChecking gradients:")
    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and name in trainable_param_names:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: gradient norm = {grad_norm:.8f}")
                if grad_norm > 1e-8:
                    grad_found = True
            else:
                print(f"  {name}: NO GRADIENT!")
    
    if not grad_found:
        print("✗ ERROR: No gradients found!")
        return False
    
    # Optimizer step
    print(f"\nPerforming optimizer step...")
    optimizer.step()
    
    # Check parameter updates
    print(f"\nChecking parameter updates:")
    updates_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and name in trainable_param_names and name in initial_params:
            param_diff = torch.abs(param.data - initial_params[name]).max().item()
            print(f"  {name}: max change = {param_diff:.8f}")
            
            if param_diff > 1e-8:
                updates_found = True
                print(f"    ✓ Parameter updated!")
            else:
                print(f"    ✗ Parameter NOT updated!")
    
    if updates_found:
        print(f"\n✓ SUCCESS: Parameters are being updated correctly!")
        return True
    else:
        print(f"\n✗ FAILURE: Parameters are NOT being updated!")
        return False

# -----------------------------------------------------------
# 6 ▸ Trainer subclass – prefix-mask & token-level CE loss
# -----------------------------------------------------------
class PromptTrainer(Trainer):
    def __init__(self, *a, prompt_types=None, trainable_params=None, **kw):
        super().__init__(*a, **kw)
        self.prompt_types = prompt_types
        self.trainable_params = trainable_params or []
        self._gradient_check_done = False
        self._first_step_done = False
        self._param_values_before = {}
        self.train_sampler = None  # Will be set externally
        
    def get_train_dataloader(self):
        """Override to use weighted sampler if available"""
        if self.train_sampler is not None:
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self.train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # Fall back to default behavior
            return super().get_train_dataloader()
        
    def training_step(self, model, inputs):
        """Override training step to check parameter updates"""
        
        # Store parameter values before first step
        if not self._first_step_done:
            print("\n" + "="*60)
            print("CHECKING PARAMETER UPDATE AFTER ONE BACKPROPAGATION")
            print("="*60)
            
            # Verify optimizer configuration now that it's initialized
            print(f"\nOptimizer configuration:")
            print(f"  Type: {type(self.optimizer)}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr}")
            
            # Check if learning rate is 0 due to warmup and fix it
            if current_lr == 0.0:
                print(f"  ⚠️  WARNING: Learning rate is 0.0! This is likely due to warmup.")
                print(f"  Setting learning rate to base LR: {self.args.learning_rate}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.args.learning_rate
                print(f"  ✅ Learning rate corrected to: {self.args.learning_rate}")
            
            print(f"  Number of parameter groups: {len(self.optimizer.param_groups)}")
            
            # Count parameters in optimizer
            optimizer_param_count = 0
            total_params_in_optimizer = 0
            for group in self.optimizer.param_groups:
                optimizer_param_count += len(group['params'])
                for param in group['params']:
                    total_params_in_optimizer += param.numel()
            print(f"  Parameter tensors in optimizer: {optimizer_param_count}")
            print(f"  Total parameters in optimizer: {total_params_in_optimizer:,}")
            
            # Verify that all trainable parameters are in the optimizer
            trainable_model_params = set()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_model_params.add(id(param))
            
            optimizer_params = set()
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    optimizer_params.add(id(param))
            
            if trainable_model_params == optimizer_params:
                print("  ✓ All trainable model parameters are in the optimizer")
            else:
                print("  ✗ WARNING: Mismatch between trainable model parameters and optimizer parameters!")
                missing_in_optimizer = trainable_model_params - optimizer_params
                extra_in_optimizer = optimizer_params - trainable_model_params
                print(f"    Missing from optimizer: {len(missing_in_optimizer)} parameters")
                print(f"    Extra in optimizer: {len(extra_in_optimizer)} parameters")
            
            # Store initial parameter values
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.trainable_params:
                    self._param_values_before[name] = param.data.clone()
                    print(f"Storing initial value for: {name}")
            
            print(f"Ready to test parameter updates...")
        
        # Perform the actual training step
        result = super().training_step(model, inputs)
        
        # Check parameter updates after first step
        if not self._first_step_done:
            print("\nChecking parameter updates after backpropagation:")
            updates_found = False
            
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.trainable_params:
                    if name in self._param_values_before:
                        param_diff = torch.abs(param.data - self._param_values_before[name]).max().item()
                        print(f"  {name}: max_change = {param_diff:.8f}")
                        
                        if param_diff > 1e-8:  # Threshold for detecting changes
                            updates_found = True
                            print(f"    ✓ Parameter updated!")
                        else:
                            print(f"    ✗ Parameter NOT updated!")
                        
                        # Show gradient info
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print(f"    Gradient norm: {grad_norm:.8f}")
                        else:
                            print(f"    ✗ No gradient found!")
            
            if updates_found:
                print("\n✓ SUCCESS: Parameters are being updated!")
            else:
                print("\n✗ ERROR: No parameter updates detected!")
                print("This indicates the optimizer is not working properly.")
                
                # Debug the optimizer step in detail
                print("\n--- DEBUGGING OPTIMIZER STEP ---")
                
                # Check if gradients are being clipped too much
                total_norm = 0.0
                param_count = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                total_norm = total_norm ** (1. / 2)
                print(f"Total gradient norm: {total_norm:.6f}")
                print(f"Parameters with gradients: {param_count}")
                
                # Check optimizer state
                print(f"Optimizer state_dict keys: {list(self.optimizer.state_dict().keys())}")
                
                # Check if this is a gradient clipping issue
                if hasattr(self.args, 'max_grad_norm'):
                    print(f"Max grad norm setting: {self.args.max_grad_norm}")
                
                # Check parameter scaling (fp16 related)
                if hasattr(self.optimizer, 'scaler') and self.optimizer.scaler is not None:
                    print(f"Gradient scaler scale: {self.optimizer.scaler.get_scale()}")
                else:
                    print("No gradient scaler found (fp16 disabled)")
                
                # Check if gradients are being clipped too aggressively
                if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm is not None:
                    print(f"Gradient clipping threshold: {self.args.max_grad_norm}")
                    if total_norm > self.args.max_grad_norm:
                        clipping_ratio = self.args.max_grad_norm / total_norm
                        print(f"⚠️  Gradients are being clipped! Ratio: {clipping_ratio:.6f}")
                        print(f"Original norm: {total_norm:.6f}, Clipped to: {self.args.max_grad_norm}")
                    else:
                        print(f"✓ Gradients within clipping threshold")
                else:
                    print("No gradient clipping enabled")
                
                # CRITICAL DEBUG: Test if the accelerated optimizer is the issue
                print("\n--- TESTING VANILLA PYTORCH OPTIMIZER ---")
                
                # Create a vanilla PyTorch optimizer with the same parameters
                trainable_params_list = [p for name, p in model.named_parameters() 
                                       if param.requires_grad and name in self.trainable_params]
                vanilla_optimizer = torch.optim.AdamW(trainable_params_list, lr=5e-5)
                
                # Store current parameter values
                current_param_values = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.trainable_params:
                        current_param_values[name] = param.data.clone()
                
                # Apply one step with vanilla optimizer using existing gradients
                vanilla_optimizer.step()
                
                # Check if vanilla optimizer updated parameters
                vanilla_updates_found = False
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.trainable_params:
                        if name in current_param_values:
                            param_diff = torch.abs(param.data - current_param_values[name]).max().item()
                            if param_diff > 1e-8:
                                vanilla_updates_found = True
                                print(f"  ✓ Vanilla optimizer updated {name}: max_change = {param_diff:.8f}")
                                break
                
                if vanilla_updates_found:
                    print("  ✅ SUCCESS: Vanilla PyTorch optimizer works!")
                    print("  🔍 DIAGNOSIS: The issue is with accelerate.optimizer.AcceleratedOptimizer")
                else:
                    print("  ❌ FAILURE: Even vanilla optimizer doesn't work - deeper issue")
                
                print("Checking one parameter in detail:")
                first_param_name = self.trainable_params[0]
                for name, param in model.named_parameters():
                    if name == first_param_name:
                        print(f"  Parameter: {name}")
                        print(f"  Shape: {param.shape}")
                        print(f"  Gradient shape: {param.grad.shape if param.grad is not None else 'None'}")
                        print(f"  Gradient mean: {param.grad.mean().item() if param.grad is not None else 'None'}")
                        print(f"  Gradient std: {param.grad.std().item() if param.grad is not None else 'None'}")
                        print(f"  Parameter mean before: {self._param_values_before[name].mean().item()}")
                        print(f"  Parameter mean after: {param.data.mean().item()}")
                        print(f"  Parameter std before: {self._param_values_before[name].std().item()}")
                        print(f"  Parameter std after: {param.data.std().item()}")
                        break
                
            # Also check optimizer state
            if hasattr(self, 'optimizer'):
                print(f"\nOptimizer info:")
                print(f"  Type: {type(self.optimizer)}")
                print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']}")
                print(f"  Number of parameter groups: {len(self.optimizer.param_groups)}")
                
                total_optimizer_params = sum(len(group['params']) for group in self.optimizer.param_groups)
                print(f"  Total parameters in optimizer: {total_optimizer_params}")
            
            self._first_step_done = True
            print("="*60)
            
        return result
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # One-time gradient check on first call
        if not self._gradient_check_done:
            trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
            if not trainable_params:
                print("CRITICAL ERROR: No trainable parameters in compute_loss!")
                raise RuntimeError("No trainable parameters found during training!")
            print(f"Verified {len(trainable_params)} trainable parameters in trainer")
            
            # Verify these match our expected trainable parameters
            expected_set = set(self.trainable_params)
            actual_set = set(trainable_params)
            if expected_set != actual_set:
                print(f"WARNING: Mismatch between expected and actual trainable parameters!")
                print(f"Expected: {expected_set}")
                print(f"Actual: {actual_set}")
            
            self._gradient_check_done = True
        
        # Filter out unwanted keys that might cause issues with the model forward method
        # Keep only the keys that the model expects
        model_input_keys = {'input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 
                           'head_mask', 'inputs_embeds', 'labels', 'output_attentions', 
                           'output_hidden_states', 'return_dict'}
        
        filtered_inputs = {k: v for k, v in inputs.items() if k in model_input_keys}
        
        out = model(**filtered_inputs)
        
        # Simplified and more effective loss calculation
        labels_flat = filtered_inputs["labels"].view(-1)
        valid_mask = labels_flat != -100
        
        if valid_mask.sum() > 0:
            # Use simple CrossEntropyLoss with label smoothing for better generalization
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(out.logits.view(-1, model.num_labels), labels_flat)
            
            # Add a small amount of L2 regularization on the logits to prevent overconfidence
            logits_reg = 0.001 * torch.mean(out.logits ** 2)
            loss = loss + logits_reg
            
            # Optional: Add focal loss component if accuracy is very low
            if hasattr(self, '_training_step_count'):
                self._training_step_count += 1
            else:
                self._training_step_count = 1
                
            # After some steps, check if we need focal loss
            # if self._training_step_count > 0 and self._training_step_count % 30 == 0:
            #     print('use focal loss')
            #     focal_loss_fct = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=-100)
            #     focal_loss = focal_loss_fct(out.logits.view(-1, model.num_labels), labels_flat)
            #     loss = 0.8 * loss + 0.2 * focal_loss  # Mix in focal loss for hard examples
                
        else:
            # Fallback to standard loss if no valid labels
            # loss_fct = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=-100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(out.logits.view(-1, model.num_labels), labels_flat)
            
        return (loss, out) if return_outputs else loss
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Override optimizer and scheduler creation to completely bypass Accelerate"""
        print("\n🔧 OVERRIDING OPTIMIZER AND SCHEDULER CREATION TO BYPASS ACCELERATE")
        
        # Get all trainable parameters
        decay_parameters = []
        no_decay_parameters = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Apply weight decay to weights but not biases or layer norms
                if any(nd in name for nd in ["bias", "layer_norm", "layernorm"]):
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
        
        # Create parameter groups
        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]
        
        # Create vanilla PyTorch optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        print(f"✅ Created vanilla PyTorch AdamW optimizer")
        print(f"  Parameters with decay: {len(decay_parameters):,}")
        print(f"  Parameters without decay: {len(no_decay_parameters):,}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Weight decay: {self.args.weight_decay}")
        
        # Create scheduler
        from transformers.optimization import get_scheduler
        
        self.lr_scheduler = get_scheduler(
            name="linear",  # Use linear decay after warmup
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        print(f"✅ Created learning rate scheduler")
        print(f"  Type: linear warmup + decay")
        print(f"  Warmup steps: {self.args.warmup_steps}")
        print(f"  Total training steps: {num_training_steps}")
        
        # CRITICAL: Set flags to prevent Accelerate from wrapping our optimizer
        # This is a hack but necessary to bypass Accelerate
        self._created_lr_scheduler = True
        
        # Store the original optimizer to prevent it from being wrapped
        self._vanilla_optimizer = self.optimizer
        
        return self.optimizer, self.lr_scheduler
    
    def optimizer_step(self, optimizer):
        """Override optimizer step to ensure we use our vanilla optimizer"""
        # Use our stored vanilla optimizer instead of the potentially wrapped one
        if hasattr(self, '_vanilla_optimizer'):
            actual_optimizer = self._vanilla_optimizer
            print(f"🔧 Using vanilla optimizer for step: {type(actual_optimizer)}")
        else:
            actual_optimizer = optimizer
            print(f"⚠️  Using provided optimizer: {type(actual_optimizer)}")
        
        # Perform the optimizer step
        actual_optimizer.step()
        
        # Also step the scheduler if it exists
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Clear gradients
        actual_optimizer.zero_grad()
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """Override the entire training loop to use vanilla PyTorch optimizer"""
        print("\n🔧 USING CUSTOM TRAINING LOOP TO BYPASS ACCELERATE")
        
        # Initialize basic training setup
        self._train_batch_size = batch_size
        
        model = self._wrap_model(self.model_wrapped)
        
        # Use only our vanilla optimizer without any Accelerate wrapping
        if not hasattr(self, '_vanilla_optimizer'):
            # Create vanilla optimizer if not exists
            decay_parameters = []
            no_decay_parameters = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if any(nd in name for nd in ["bias", "layer_norm", "layernorm"]):
                        no_decay_parameters.append(param)
                    else:
                        decay_parameters.append(param)
            
            optimizer_grouped_parameters = [
                {"params": decay_parameters, "weight_decay": self.args.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ]
            
            self._vanilla_optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            print(f"✅ Created vanilla optimizer in training loop")
        
        # Create learning rate scheduler
        if not hasattr(self, '_vanilla_scheduler'):
            from transformers.optimization import get_scheduler
            num_training_steps = len(self.get_train_dataloader()) * self.args.num_train_epochs
            
            self._vanilla_scheduler = get_scheduler(
                name="linear",
                optimizer=self._vanilla_optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
            print(f"✅ Created vanilla scheduler in training loop")
        
        # Set model to training mode
        model.train()
        
        # Get training dataloader
        train_dataloader = self.get_train_dataloader()
        
        # Training loop
        print(f"🚀 Starting vanilla training loop with {len(train_dataloader)} steps per epoch")
        
        global_step = 0
        epoch = 0
        
        for epoch in range(int(self.args.num_train_epochs)):
            print(f"\n--- EPOCH {epoch + 1}/{int(self.args.num_train_epochs)} ---")
            
            epoch_loss = 0
            steps_in_epoch = 0
            
            for step, inputs in enumerate(train_dataloader):
                # Move inputs to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.args.device)
                
                # Forward pass
                model.train()
                outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Backward pass
                loss.backward()
                
                # Accumulate gradients
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Update parameters with our vanilla optimizer
                    self._vanilla_optimizer.step()
                    self._vanilla_scheduler.step()
                    self._vanilla_optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log progress
                    if global_step % self.args.logging_steps == 0:
                        current_lr = self._vanilla_scheduler.get_last_lr()[0]
                        print(f"Step {global_step}: loss={loss.item():.4f}, lr={current_lr:.2e}")
                        
                        # Verify parameter updates on first step
                        if global_step == 1:
                            print("🔍 Verifying parameter updates after first step...")
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in self.trainable_params[:3]:  # Check first 3
                                    if name in self._param_values_before:
                                        param_diff = torch.abs(param.data - self._param_values_before[name]).max().item()
                                        if param_diff > 1e-8:
                                            print(f"  ✅ {name}: updated by {param_diff:.8f}")
                                        else:
                                            print(f"  ❌ {name}: no change detected")
                
                epoch_loss += loss.item()
                steps_in_epoch += 1
                
                # Break if max steps reached
                if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    break
            
            avg_epoch_loss = epoch_loss / steps_in_epoch
            print(f"Epoch {epoch + 1} completed: avg_loss={avg_epoch_loss:.4f}")
            
            # Evaluation
            if self.args.evaluation_strategy == "epoch":
                eval_results = self.evaluate()
                print(f"Evaluation results: {eval_results}")
        
        print(f"✅ Training completed! Total steps: {global_step}")
        return None  # Return None to indicate successful completion

# -----------------------------------------------------------
# 7 ▸ Metrics - Enhanced with clear balanced accuracy reporting
# -----------------------------------------------------------
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def compute_metrics(pred):
    lbls = pred.label_ids.reshape(-1)
    prds = pred.predictions.argmax(-1).reshape(-1)
    mask = lbls != -100
    lbls, prds = lbls[mask], prds[mask]
    
    # Calculate both regular and balanced accuracy
    regular_accuracy = accuracy_score(lbls, prds)
    balanced_acc = balanced_accuracy_score(lbls, prds)
    
    return {
        "accuracy": regular_accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision_score(lbls, prds, average="macro"),
        "recall": recall_score(lbls, prds, average="macro"),
        "f1": f1_score(lbls, prds, average="macro"),
    }

def analyze_class_distribution_and_predictions(pred, class_id_dict, id_class_dict, dataset_name=""):
    """Analyze class distribution and model predictions to understand balanced accuracy issues"""
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION & PREDICTION ANALYSIS - {dataset_name}")
    print(f"{'='*60}")
    
    lbls = pred.label_ids.reshape(-1)
    prds = pred.predictions.argmax(-1).reshape(-1)
    mask = lbls != -100
    lbls, prds = lbls[mask], prds[mask]
    
    # Get unique classes present in the data
    unique_labels = np.unique(lbls)
    unique_predictions = np.unique(prds)
    
    print(f"Total samples (excluding -100): {len(lbls)}")
    print(f"Unique classes in labels: {len(unique_labels)} -> {unique_labels}")
    print(f"Unique classes in predictions: {len(unique_predictions)} -> {unique_predictions}")
    
    # Class distribution analysis
    print(f"\nCLASS DISTRIBUTION IN LABELS:")
    label_counts = {}
    for label in unique_labels:
        count = np.sum(lbls == label)
        percentage = 100 * count / len(lbls)
        class_name = id_class_dict.get(label, f"Unknown_{label}")
        label_counts[label] = count
        print(f"  Class {label} ({class_name}): {count:,} samples ({percentage:.2f}%)")
    
    # Prediction distribution analysis
    print(f"\nCLASS DISTRIBUTION IN PREDICTIONS:")
    pred_counts = {}
    for pred_class in unique_predictions:
        count = np.sum(prds == pred_class)
        percentage = 100 * count / len(prds)
        class_name = id_class_dict.get(pred_class, f"Unknown_{pred_class}")
        pred_counts[pred_class] = count
        print(f"  Class {pred_class} ({class_name}): {count:,} predictions ({percentage:.2f}%)")
    
    # Per-class accuracy analysis
    print(f"\nPER-CLASS PERFORMANCE:")
    class_accuracies = []
    for label in unique_labels:
        mask_class = lbls == label
        if np.sum(mask_class) > 0:
            class_correct = np.sum((lbls == label) & (prds == label))
            class_total = np.sum(mask_class)
            class_accuracy = class_correct / class_total
            class_accuracies.append(class_accuracy)
            
            class_name = id_class_dict.get(label, f"Unknown_{label}")
            print(f"  Class {label} ({class_name}): {class_correct}/{class_total} = {class_accuracy:.4f} ({100*class_accuracy:.2f}%)")
        else:
            print(f"  Class {label}: No samples in this set")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    cm = confusion_matrix(lbls, prds, labels=unique_labels)
    print("     Predicted:")
    print("     ", end="")
    for pred_class in unique_labels:
        print(f"{pred_class:>8}", end="")
    print()
    
    for i, true_class in enumerate(unique_labels):
        class_name = id_class_dict.get(true_class, f"Unk_{true_class}")
        print(f"T{true_class:2} ({class_name[:4]:>4})", end="")
        for j, pred_class in enumerate(unique_labels):
            print(f"{cm[i,j]:>8}", end="")
        print()
    
    # Calculate regular vs balanced accuracy
    regular_accuracy = accuracy_score(lbls, prds)
    balanced_accuracy = balanced_accuracy_score(lbls, prds)
    
    print(f"\nACCURACY ANALYSIS:")
    print(f"  Regular Accuracy: {regular_accuracy:.4f} ({100*regular_accuracy:.2f}%)")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f} ({100*balanced_accuracy:.2f}%)")
    print(f"  Difference: {regular_accuracy - balanced_accuracy:.4f}")
    
    # Identify most/least accurate classes
    if len(class_accuracies) > 1:
        min_acc_idx = np.argmin(class_accuracies)
        max_acc_idx = np.argmax(class_accuracies)
        min_class = unique_labels[min_acc_idx]
        max_class = unique_labels[max_acc_idx]
        
        print(f"\n  Best performing class: {max_class} ({id_class_dict.get(max_class, 'Unknown')}) = {class_accuracies[max_acc_idx]:.4f}")
        print(f"  Worst performing class: {min_class} ({id_class_dict.get(min_class, 'Unknown')}) = {class_accuracies[min_acc_idx]:.4f}")
        
        # Check if model is biased toward majority class
        majority_class = max(label_counts.keys(), key=lambda x: label_counts[x])
        majority_pred_count = pred_counts.get(majority_class, 0)
        total_preds = len(prds)
        
        print(f"\n  Majority class: {majority_class} ({id_class_dict.get(majority_class, 'Unknown')})")
        print(f"  Majority class in labels: {label_counts[majority_class]} ({100*label_counts[majority_class]/len(lbls):.1f}%)")
        print(f"  Majority class in predictions: {majority_pred_count} ({100*majority_pred_count/total_preds:.1f}%)")
        
        if majority_pred_count / total_preds > 0.8:
            print(f"  ⚠️  WARNING: Model appears to be heavily biased toward majority class!")
            print(f"      This explains why regular accuracy is decent but balanced accuracy is poor.")
            print(f"      The model is mostly predicting the majority class.")
    
    print(f"{'='*60}")
    
    return {
        'regular_accuracy': regular_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'class_accuracies': dict(zip(unique_labels, class_accuracies)) if class_accuracies else {},
        'label_distribution': label_counts,
        'prediction_distribution': pred_counts,
        'confusion_matrix': cm.tolist(),
    }
# -----------------------------------------------------------
# 8 ▸ K-Fold Cross Validation
# -----------------------------------------------------------

# Create output directory
base_run_dir = Path(args.output_root) / "5fold_cv" / Path(args.dataset_file).stem / datetime.datetime.now().strftime("%y%m%d_%H%M%S")
base_run_dir.mkdir(parents=True, exist_ok=True)

# Initialize regular KFold (no stratification needed with weighted sampling)
from sklearn.model_selection import KFold
import torch.utils.data

def get_sample_majority_class(sample):
    """Get the majority class for a sample to use for weighted sampling"""
    labels = np.array(sample["labels"])
    valid_labels = labels[labels != -100]
    if len(valid_labels) == 0:
        return 0  # Default class if no valid labels
    # Return the most frequent class in this sample
    unique, counts = np.unique(valid_labels, return_counts=True)
    return unique[np.argmax(counts)]

def get_weighted_sampler(dataset, data_global_describe=None):
    """Create a weighted sampler for balanced training"""
    print("Creating weighted sampler for balanced training...")
    
    # Get majority class for each sample
    sample_classes = []
    for i in range(len(dataset)):
        majority_class = get_sample_majority_class(dataset[i])
        sample_classes.append(majority_class)
    
    sample_classes = np.array(sample_classes)
    
    # Calculate class counts and weights
    unique_classes, class_counts = np.unique(sample_classes, return_counts=True)
    print(f"Class distribution in dataset:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples ({100*count/len(sample_classes):.1f}%)")
    
    # Create class weights (inverse frequency)
    class_weight_dict = {}
    for cls, count in zip(unique_classes, class_counts):
        class_weight_dict[cls] = 1.0 / count
    
    # Assign weights to each sample
    sample_weights = np.array([class_weight_dict[cls] for cls in sample_classes])
    
    # Normalize weights
    sample_weights = sample_weights / np.sum(sample_weights)
    
    print(f"Sample weights range: {sample_weights.min():.6f} to {sample_weights.max():.6f}")
    
    # Create weighted sampler
    num_samples = data_global_describe.get("num_training_cells", len(dataset)) if data_global_describe else len(dataset)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples,
        replacement=True
    )
    
    print(f"Created weighted sampler with {num_samples} samples per epoch")
    return train_sampler

# Create simple class distribution for reference
sample_classes = []
for i in range(len(filtered_ds)):
    majority_class = get_sample_majority_class(filtered_ds[i])
    sample_classes.append(majority_class)

print(f"Overall class distribution:")
unique_classes, class_counts = np.unique(sample_classes, return_counts=True)
for cls, count in zip(unique_classes, class_counts):
    print(f"  Class {cls}: {count} samples ({100*count/len(sample_classes):.1f}%)")

kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

# Store results for all folds
all_fold_results = []
fold_test_metrics = []

print(f"\nStarting {args.n_folds}-fold cross validation...")
print(f"Total samples: {len(filtered_ds)}")

# Convert dataset to indices for KFold
dataset_indices = list(range(len(filtered_ds)))

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset_indices)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{args.n_folds}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    # Clear GPU cache before each fold
    torch.cuda.empty_cache()
    
    # Create train and test datasets for this fold
    train_ds = filtered_ds.select(train_idx)
    test_ds = filtered_ds.select(test_idx)
    
    # Split training data to get validation set (80/20 split of training data)
    train_val_split = train_ds.train_test_split(test_size=0.2, seed=args.seed + fold)
    train_fold_ds = train_val_split["train"]
    eval_fold_ds = train_val_split["test"]
    
    print(f"Final split - Train: {len(train_fold_ds)}, Val: {len(eval_fold_ds)}, Test: {len(test_ds)}")
    
    # Analyze class distribution in this fold's data
    def analyze_fold_class_distribution(dataset, set_name):
        all_labels = []
        for sample in dataset:
            labels = sample["labels"]
            valid_labels = [l for l in labels if l != -100]
            all_labels.extend(valid_labels)
        
        if all_labels:
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"{set_name} class distribution:")
            total = len(all_labels)
            for cls, count in zip(unique, counts):
                class_name = id_class_dict.get(cls, f"Unknown_{cls}")
                print(f"  Class {cls} ({class_name}): {count:,} labels ({100*count/total:.1f}%)")
        else:
            print(f"{set_name}: No valid labels found!")
    
    analyze_fold_class_distribution(train_fold_ds, "Training set")
    analyze_fold_class_distribution(eval_fold_ds, "Validation set") 
    analyze_fold_class_distribution(test_ds, "Test set")
    
    # Detailed data inspection
    inspect_training_data(train_fold_ds, class_id_dict, id_class_dict, f"FOLD {fold + 1} TRAINING")
    
    # Create fresh model for this fold using the enhanced function
    model, prompt_types, trainable_param_names = create_model()
    
    # Verify trainable parameters one more time
    trainable_count, total_count = count_trainable(model, trainable_param_names)
    
    if trainable_count == 0:
        raise RuntimeError("ERROR: create_model() still returned no trainable parameters!")
    
    # Test parameter updates with a single forward/backward pass
    print(f"\nTesting parameter updates before starting training...")
    update_test_passed = test_parameter_updates(model, data_collator, train_fold_ds, trainable_param_names)
    
    if not update_test_passed:
        print("ERROR: Parameter update test failed! Aborting training for this fold.")
        continue  # Skip to next fold
    
    print("✓ Parameter update test passed! Proceeding with training.")
    
    # Warm up the model before training
    model = warm_up_model(model, data_collator, train_fold_ds)
    
    # Create fold-specific output directory
    fold_dir = base_run_dir / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(fold_dir)
    
    # Training arguments for this fold with improved convergence settings
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,  # Reduced for more frequent updates
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",  # Evaluate more frequently
        eval_steps=50,  # Evaluate every 50 steps
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_balanced_accuracy",
        greater_is_better=True,
        warmup_steps=50,  # Increased warmup for stability
        warmup_ratio=0.1,  # Higher warmup ratio
        weight_decay=0.01,
        report_to="none",
        logging_dir=str(fold_dir / "logs"),
        gradient_checkpointing=False,
        fp16=False,  # Keep disabled for stability
        max_grad_norm=0.5,  # Reduced gradient clipping for better convergence
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        logging_steps=10,  # Log every 10 steps for better monitoring
        no_cuda=False,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,  # Drop last incomplete batch for stability
        local_rank=-1,
        deepspeed=None,
        label_smoothing_factor=0.1,  # Add label smoothing for better generalization
        lr_scheduler_type="cosine",  # Use cosine annealing for better convergence
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        optim="adamw_torch",  # Force use of PyTorch AdamW
    )
    
    # Create weighted sampler for balanced training
    data_global_describe = {
        "num_training_cells": len(train_fold_ds)
    }
    train_sampler = get_weighted_sampler(train_fold_ds, data_global_describe)
    
    # Create trainer for this fold with enhanced early stopping and learning rate management
    trainer = PromptTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_fold_ds,
        eval_dataset=eval_fold_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=10),  # Increased patience
            LearningRateResetCallback(patience=100, factor=0.7, min_lr=1e-6)  # LR reset for plateau
        ],
        prompt_types=prompt_types,
        trainable_params=trainable_param_names,
    )
    
    # Override the default data loader with weighted sampler
    trainer.train_sampler = train_sampler
    print(f"✅ Weighted sampler configured for balanced training in fold {fold + 1}")
    
    # Start training - optimizer verification will happen in PromptTrainer.training_step
    print(f"Training fold {fold + 1}...")
    trainer.train()
    
    # Evaluate on test set
    print(f"Evaluating fold {fold + 1}...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)
    trainer.save_model(fold_dir)
    
    # Get predictions for detailed analysis
    print(f"Getting predictions for detailed analysis...")
    test_predictions = trainer.predict(test_ds)
    
    # Analyze class distribution and predictions to understand low balanced accuracy
    analyze_class_distribution_and_predictions(
        test_predictions, 
        class_id_dict, 
        id_class_dict, 
        dataset_name=f"Test Set Fold {fold + 1}"
    )
    
    # Store results
    fold_results = {
        'fold': fold + 1,
        'test_accuracy': test_metrics.get('test_accuracy', 0),
        'test_balanced_accuracy': test_metrics.get('test_balanced_accuracy', 0),
        'test_precision': test_metrics.get('test_precision', 0),
        'test_recall': test_metrics.get('test_recall', 0),
        'test_f1': test_metrics.get('test_f1', 0),
        'train_samples': len(train_fold_ds),
        'val_samples': len(eval_fold_ds),
        'test_samples': len(test_ds)
    }
    
    all_fold_results.append(fold_results)
    fold_test_metrics.append(test_metrics)
    
    # Display fold results
    print(f"\nFOLD {fold + 1} TEST RESULTS:")
    print(f"Regular Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"Balanced Accuracy: {test_metrics.get('test_balanced_accuracy', 'N/A'):.4f}")
    print(f"Precision (macro): {test_metrics.get('test_precision', 'N/A'):.4f}")
    print(f"Recall (macro): {test_metrics.get('test_recall', 'N/A'):.4f}")
    print(f"F1 Score (macro): {test_metrics.get('test_f1', 'N/A'):.4f}")
    
    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

# -----------------------------------------------------------
# 9 ▸ Aggregate results across all folds
# -----------------------------------------------------------
print(f"\n{'='*80}")
print("5-FOLD CROSS VALIDATION SUMMARY (WITH WEIGHTED SAMPLING)")
print("Note: Using WeightedRandomSampler instead of stratified splits for class balance")
print(f"{'='*80}")

# Calculate mean and std for each metric
metrics_to_aggregate = ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 'test_recall', 'test_f1']
aggregated_results = {}

for metric in metrics_to_aggregate:
    values = [fold[metric] for fold in all_fold_results]
    aggregated_results[metric] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'values': values
    }

# Display aggregated results
print(f"Regular Accuracy: {aggregated_results['test_accuracy']['mean']:.4f} ± {aggregated_results['test_accuracy']['std']:.4f}")
print(f"Balanced Accuracy: {aggregated_results['test_balanced_accuracy']['mean']:.4f} ± {aggregated_results['test_balanced_accuracy']['std']:.4f}")
print(f"Precision (macro): {aggregated_results['test_precision']['mean']:.4f} ± {aggregated_results['test_precision']['std']:.4f}")
print(f"Recall (macro): {aggregated_results['test_recall']['mean']:.4f} ± {aggregated_results['test_recall']['std']:.4f}")
print(f"F1 Score (macro): {aggregated_results['test_f1']['mean']:.4f} ± {aggregated_results['test_f1']['std']:.4f}")

print(f"\nPer-fold results:")
for i, fold_result in enumerate(all_fold_results):
    print(f"Fold {i+1}: Acc={fold_result['test_accuracy']:.4f}, "
          f"Bal_Acc={fold_result['test_balanced_accuracy']:.4f}, "
          f"F1={fold_result['test_f1']:.4f}")

# Save comprehensive results
final_results = {
    'aggregated_results': aggregated_results,
    'fold_results': all_fold_results,
    'experiment_config': {
        'n_folds': args.n_folds,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'dataset_file': args.dataset_file,
        'total_samples': len(filtered_ds)
    }
}

results_file = base_run_dir / "5fold_cv_results.json"
with open(results_file, 'w') as f:
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json_results = json.loads(json.dumps(final_results, default=convert_numpy))
    json.dump(json_results, f, indent=2)

print(f"\nDetailed results saved to: {results_file}")
print(f"Individual fold results saved in: {base_run_dir}")
print(f"{'='*80}")

# -----------------------------------------------------------
# 10 ▸ Learning Rate Monitoring and Reset Callback
# -----------------------------------------------------------
from transformers import TrainerCallback





