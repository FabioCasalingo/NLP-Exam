"""
model_manager.py - Handles transformer model loading and saving

This module provides a unified interface for working with HuggingFace transformer models,
specifically focused on masked language models like RoBERTa. It handles model loading from
the HuggingFace Hub or local disk, saving checkpoints, and various model manipulation tasks
like layer freezing and adapter integration.

The ModelManager class simplifies common workflows like:
- Loading pretrained models with proper caching
- Saving and loading fine-tuned models
- Freezing layers for efficient fine-tuning
- Managing training checkpoints
- Comparing model differences
"""

import os
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


class ModelManager:
    """
    Manages transformer models throughout the project lifecycle.
    
    This class provides a clean interface for all model-related operations, from loading
    pretrained models to saving fine-tuned versions. It automatically handles device
    placement (CPU/GPU), directory creation, and provides utilities for model inspection
    and manipulation.
    
    The manager organizes models into baseline and debiased directories, making it easy
    to compare original and modified versions.
    """
    
    def __init__(self, models_dir: str = "./models", cache_dir: Optional[str] = None):
        """
        Initialize the model manager with directory structure.
        
        Sets up the directory hierarchy for storing models and creates necessary folders.
        Also detects available hardware (CPU/GPU) and displays relevant information.
        
        Args:
            models_dir: Root directory for model storage. Will contain subdirectories for
                       baseline, debiased, and cached models.
            cache_dir: Optional custom cache directory for HuggingFace downloads. If None,
                      uses a cache subdirectory within models_dir.
        """
        self.models_dir = Path(models_dir)
        self.baseline_dir = self.models_dir / "baseline"
        self.debiased_dir = self.models_dir / "debiased"
        
        # Set up cache directory for HuggingFace downloads
        self.cache_dir = cache_dir if cache_dir else str(self.models_dir / "cache")
        
        # Create all necessary directories
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.debiased_dir.mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Detect and display device information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_pretrained_roberta(self, model_name: str = "roberta-base",
                               use_cache: bool = True) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pretrained RoBERTa model from HuggingFace Hub.
        
        This is the primary method for loading base models. It downloads from HuggingFace
        (or uses cached versions if available), moves the model to the appropriate device,
        and displays useful information about model size and parameters.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'roberta-base', 'roberta-large').
                       Defaults to 'roberta-base' which is a good balance of performance and size.
            use_cache: Whether to use the local cache for faster loading. Set to False to
                      force a fresh download. Defaults to True.
        
        Returns:
            Tuple of (model, tokenizer) ready for use. Model is already moved to the
            appropriate device (GPU if available, otherwise CPU).
        
        Raises:
            Exception: If model loading fails due to network issues, invalid model name,
                      or other errors.
        """
        print(f"\nLoading {model_name}...")
        
        cache = self.cache_dir if use_cache else None
        
        try:
            print("Loading tokenizer...")
            tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache)
            
            print("Loading model...")
            model = RobertaForMaskedLM.from_pretrained(model_name, cache_dir=cache)
            model = model.to(self.device)
            
            # Calculate parameter counts for user information
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\nModel loaded successfully")
            print(f"Total parameters: {num_params:,}")
            print(f"Trainable parameters: {num_trainable:,}")
            print(f"Model size: ~{num_params * 4 / 1e9:.2f} GB")
            print(f"Device: {self.device}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_model_from_path(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a locally saved model from disk.
        
        Use this to load models that have been fine-tuned and saved, or models downloaded
        manually. The method automatically detects the model type and loads both the model
        and its tokenizer.
        
        Args:
            model_path: Path to the directory containing the saved model and tokenizer files.
                       Should contain config.json, pytorch_model.bin, and tokenizer files.
        
        Returns:
            Tuple of (model, tokenizer) loaded from the specified path.
        
        Raises:
            FileNotFoundError: If the model path doesn't exist.
            Exception: If model loading fails due to corrupted files or other errors.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\nLoading model from {model_path}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            model = model.to(self.device)
            
            print(f"Model loaded from disk")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                  save_path: str, save_name: str = "final") -> str:
        """
        Save a model and tokenizer to disk.
        
        This saves the complete model including weights, configuration, and tokenizer.
        Also creates a metadata file with useful information about the model. The saved
        model can be reloaded later using load_model_from_path.
        
        Args:
            model: The model to save.
            tokenizer: The tokenizer to save alongside the model.
            save_path: Base directory where the model should be saved.
            save_name: Subdirectory name within save_path. Defaults to "final".
        
        Returns:
            String path to the directory where the model was saved.
        
        Raises:
            Exception: If saving fails due to disk space, permissions, or other errors.
        """
        save_dir = Path(save_path) / save_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving model to {save_dir}...")
        
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
            # Save metadata for future reference
            info = {
                'model_type': model.config.model_type,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(self.device),
                'save_path': str(save_dir)
            }
            
            import json
            with open(save_dir / "model_info.json", 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"Model saved successfully")
            print(f"Location: {save_dir}")
            
            return str(save_dir)
            
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def save_checkpoint(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                       checkpoint_name: str, metrics: Optional[Dict] = None) -> str:
        """
        Save a training checkpoint with optional metrics.
        
        Checkpoints are useful during training to save intermediate states. This method
        saves the model, tokenizer, and optionally any metrics you want to track (like
        loss, accuracy, or bias scores at this point in training).
        
        Args:
            model: The model to checkpoint.
            tokenizer: The tokenizer to save.
            checkpoint_name: Name for this checkpoint (e.g., "epoch_3" or "step_1000").
            metrics: Optional dictionary of metrics to save alongside the model.
                    Could include training loss, validation scores, bias metrics, etc.
        
        Returns:
            String path to the checkpoint directory.
        """
        checkpoint_dir = self.debiased_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving checkpoint: {checkpoint_name}...")
        
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save metrics if provided
        if metrics:
            import json
            with open(checkpoint_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_dir}")
        
        return str(checkpoint_dir)
    
    # Utility methods
    
    def get_model_info(self, model: PreTrainedModel) -> Dict:
        """
        Extract detailed information about a model.
        
        This gathers comprehensive statistics about the model including parameter counts,
        architecture details, and memory requirements. Useful for understanding model
        characteristics and comparing different models.
        
        Args:
            model: The model to analyze.
        
        Returns:
            Dictionary containing model statistics including parameter counts, architecture
            dimensions (hidden size, layers, etc.), and size estimates.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        info = {
            'model_type': model.config.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
            'model_size_gb': total_params * 4 / 1e9,  # Assuming float32
            'hidden_size': model.config.hidden_size,
            'num_layers': model.config.num_hidden_layers,
            'num_attention_heads': model.config.num_attention_heads,
            'vocab_size': model.config.vocab_size
        }
        
        return info
    
    def print_model_info(self, model: PreTrainedModel):
        """
        Display model information in a readable format.
        
        Pretty-prints the output of get_model_info for easy inspection. Useful for
        quickly understanding what you're working with.
        
        Args:
            model: The model to describe.
        """
        info = self.get_model_info(model)
        
        print("\nModel Information:")
        print(f"Type: {info['model_type']}")
        print(f"Total params: {info['total_parameters']:,}")
        print(f"Trainable: {info['trainable_parameters']:,} ({info['trainable_percentage']:.1f}%)")
        print(f"Frozen: {info['frozen_parameters']:,}")
        print(f"Size: ~{info['model_size_gb']:.2f} GB")
        print(f"\nArchitecture:")
        print(f"Hidden size: {info['hidden_size']}")
        print(f"Layers: {info['num_layers']}")
        print(f"Attention heads: {info['num_attention_heads']}")
        print(f"Vocab size: {info['vocab_size']:,}")
    
    def freeze_layers(self, model: PreTrainedModel, 
                     num_layers_to_freeze: int = 6) -> PreTrainedModel:
        """
        Freeze the bottom layers of a model for efficient fine-tuning.
        
        Freezing lower layers is a common technique in transfer learning. The intuition
        is that lower layers learn general features while upper layers learn task-specific
        features. By freezing the bottom layers, we reduce the number of parameters to
        update, making training faster and often improving generalization.
        
        For RoBERTa-base (12 layers), freezing 6 bottom layers is a good starting point.
        
        Args:
            model: The model to modify (modified in-place).
            num_layers_to_freeze: Number of bottom encoder layers to freeze. Also freezes
                                 the embedding layer. Defaults to 6.
        
        Returns:
            The modified model with frozen layers. Note that the model is modified in-place,
            so the return value is just for convenience.
        """
        print(f"\nFreezing bottom {num_layers_to_freeze} layers...")
        
        # Freeze embedding layer - this learns basic token representations
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of encoder layers
        for i in range(num_layers_to_freeze):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Display results
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"Frozen layers: 0-{num_layers_to_freeze}")
        print(f"Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
        
        return model
    
    def compare_models(self, model1_path: str, model2_path: str) -> Dict:
        """
        Compare two models to measure their similarity.
        
        This loads two models and computes the average absolute difference between their
        parameters. Useful for verifying that fine-tuning actually changed the model, or
        for measuring how different two debiasing approaches are.
        
        Args:
            model1_path: Path to the first model.
            model2_path: Path to the second model.
        
        Returns:
            Dictionary containing comparison metrics including total parameter difference
            and whether the models are identical.
        """
        print(f"\nComparing models...")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        
        model1, _ = self.load_model_from_path(model1_path)
        model2, _ = self.load_model_from_path(model2_path)
        
        info1 = self.get_model_info(model1)
        info2 = self.get_model_info(model2)
        
        # Calculate parameter-wise differences
        param_diff = 0
        total_params = 0
        
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if n1 == n2:
                diff = torch.sum(torch.abs(p1 - p2)).item()
                param_diff += diff
                total_params += p1.numel()
        
        avg_diff = param_diff / total_params if total_params > 0 else 0
        
        comparison = {
            'model1_params': info1['total_parameters'],
            'model2_params': info2['total_parameters'],
            'param_difference': param_diff,
            'avg_param_difference': avg_diff,
            'models_identical': avg_diff < 1e-6  # Threshold for numerical precision
        }
        
        print(f"\nComparison Results:")
        print(f"Average parameter difference: {avg_diff:.6f}")
        print(f"Models identical: {comparison['models_identical']}")
        
        return comparison
    
    # Adapter methods (optional advanced feature)
    
    def add_adapters(self, model: PreTrainedModel, adapter_name: str = "debias_adapter",
                    reduction_factor: int = 16) -> PreTrainedModel:
        """
        Add adapter layers to the model for parameter-efficient fine-tuning.
        
        Adapters are small bottleneck layers inserted into each transformer layer. They
        allow fine-tuning with only 1-2% of the parameters, making training much faster
        and the saved models much smaller. This is an advanced technique that requires
        the adapter-transformers library.
        
        Args:
            model: The model to add adapters to.
            adapter_name: Name for this adapter configuration.
            reduction_factor: Bottleneck reduction factor. Higher values mean smaller
                            adapters but potentially less capacity. 16 is a good default.
        
        Returns:
            The model with adapters added. Only adapter parameters are trainable.
        """
        try:
            from transformers import AdapterConfig
            
            print(f"\nAdding adapters to model...")
            print(f"Adapter name: {adapter_name}")
            print(f"Reduction factor: {reduction_factor}")
            
            # Use Pfeiffer adapter configuration
            adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
            model.add_adapter(adapter_name, config=adapter_config)
            model.train_adapter(adapter_name)
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            
            print(f"Adapters added")
            print(f"Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
            
            return model
            
        except ImportError:
            print("Warning: adapter-transformers not installed")
            print("Install with: pip install adapter-transformers")
            print("Returning model without adapters")
            return model
        except Exception as e:
            print(f"Error adding adapters: {e}")
            return model
    
    def save_adapter(self, model: PreTrainedModel, adapter_name: str,
                    save_path: str) -> str:
        """
        Save only the adapter weights (not the full model).
        
        Since adapters are tiny compared to the full model (typically 1-2%), saving just
        the adapter results in much smaller files. The base model doesn't need to be saved
        again since it hasn't changed.
        
        Args:
            model: Model containing the adapter to save.
            adapter_name: Name of the adapter to save.
            save_path: Directory where the adapter should be saved.
        
        Returns:
            String path to the saved adapter directory.
        
        Raises:
            Exception: If the model doesn't have adapters or if saving fails.
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving adapter: {adapter_name}...")
        
        try:
            model.save_adapter(save_dir, adapter_name)
            print(f"Adapter saved: {save_dir}")
            return str(save_dir)
        except Exception as e:
            print(f"Error saving adapter: {e}")
            raise


# Standalone wrapper functions for backward compatibility
# These allow using the manager without explicitly creating an instance

def load_pretrained_roberta(model_name: str = "roberta-base",
                           cache_dir: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a pretrained RoBERTa model (convenience wrapper).
    
    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Optional cache directory.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    manager = ModelManager(cache_dir=cache_dir)
    return manager.load_pretrained_roberta(model_name)


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
              save_path: str) -> str:
    """
    Save a model and tokenizer (convenience wrapper).
    
    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        save_path: Where to save the model.
    
    Returns:
        Path to saved model directory.
    """
    manager = ModelManager()
    return manager.save_model(model, tokenizer, save_path)


def load_model(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model from disk (convenience wrapper).
    
    Args:
        model_path: Path to the saved model.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    manager = ModelManager()
    return manager.load_model_from_path(model_path)


if __name__ == "__main__":
    # Basic testing to verify the model manager works
    print("Testing model_manager.py\n")
    
    manager = ModelManager()
    
    print("\n" + "=" * 50)
    print("TEST 1: Loading RoBERTa-base")
    print("=" * 50)
    
    model, tokenizer = manager.load_pretrained_roberta()
    manager.print_model_info(model)
    
    print("\n" + "=" * 50)
    print("TEST 2: Model inference test")
    print("=" * 50)
    
    # Test basic masked language modeling
    text = "The nurse prepared for <mask> shift."
    inputs = tokenizer(text, return_tensors="pt").to(manager.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get top predictions for the mask token
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    mask_token_logits = predictions[0, mask_token_index, :]
    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    print(f"\nInput: {text}")
    print(f"Top 5 predictions for <mask>:")
    for i, token_id in enumerate(top_tokens, 1):
        token = tokenizer.decode([token_id])
        print(f"  {i}. {token}")
    
    print("\n" + "=" * 50)
    print("TEST 3: Save and reload model")
    print("=" * 50)
    
    save_path = manager.save_model(model, tokenizer, manager.baseline_dir, 
                                   save_name="test_checkpoint")
    
    print(f"\nReloading from {save_path}...")
    reloaded_model, reloaded_tokenizer = manager.load_model_from_path(save_path)
    
    print("Model reloaded successfully")
    
    print("\n" + "=" * 50)
    print("All tests passed")
    print("=" * 50)