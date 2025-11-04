"""
training.py - MLM fine-tuning for model debiasing

This module implements masked language model (MLM) training for debiasing transformer models.
It provides both a modern Trainer-based approach and a simple manual training loop, with
extensive compatibility checks to work across different versions of the transformers library.

The training process uses continued pretraining on a balanced corpus of stereotyped and
anti-stereotyped examples. By exposing the model to both types of content equally, we aim
to reduce stereotypical associations without degrading overall language modeling capability.
"""

# Environment configuration to avoid TensorFlow initialization issues
import os
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from inspect import signature

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Early stopping callback may not exist in older versions
try:
    from transformers import EarlyStoppingCallback
    HAS_EARLY_STOP = True
except Exception:
    EarlyStoppingCallback = None
    HAS_EARLY_STOP = False


def _supports_arg(obj, arg_name: str) -> bool:
    """
    Check if a class constructor supports a specific argument.
    
    This is used for compatibility checking across different transformers versions.
    Older versions may not support features like evaluation_strategy or callbacks.
    
    Args:
        obj: The class to inspect.
        arg_name: Name of the parameter to check for.
    
    Returns:
        True if the parameter is supported, False otherwise.
    """
    try:
        return arg_name in signature(obj.__init__).parameters
    except Exception:
        return False


class TextDataset(Dataset):
    """
    Simple dataset wrapper for text sequences.
    
    Takes a list of raw text strings and prepares them for model input by tokenizing
    and padding to a fixed length. This is used for both training and validation data.
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        """
        Initialize the text dataset.
        
        Args:
            texts: List of text strings to include in the dataset.
            tokenizer: HuggingFace tokenizer for text encoding.
            max_length: Maximum sequence length. Sequences longer than this will be
                       truncated, shorter ones will be padded. Defaults to 128.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single tokenized example.
        
        Args:
            idx: Index of the example to retrieve.
        
        Returns:
            Dictionary containing input_ids and attention_mask tensors.
        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class MLMTrainer:
    """
    Trainer for masked language model fine-tuning.
    
    This class handles the complete training workflow for debiasing via continued pretraining.
    It supports both the modern HuggingFace Trainer API (with automatic evaluation, early
    stopping, and checkpointing) and a simple manual training loop for maximum compatibility.
    
    The trainer automatically detects which features are available in your transformers
    version and adapts accordingly, ensuring the code works across different library versions.
    """

    def __init__(self, output_dir: str = "./models/debiased", device: Optional[str] = None):
        """
        Initialize the MLM trainer.
        
        Args:
            output_dir: Directory where model checkpoints and logs will be saved.
            device: Device to use for training ('cuda' or 'cpu'). If None, automatically
                   selects CUDA if available.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Detect available features in the current transformers version
        self.SUPPORTS_EVAL_STRATEGY = _supports_arg(TrainingArguments, "evaluation_strategy")
        self.SUPPORTS_GREATER_IS_BETTER = _supports_arg(TrainingArguments, "greater_is_better")
        self.SUPPORTS_CALLBACKS = _supports_arg(Trainer, "callbacks")

        print(f"MLM Trainer initialized on {self.device}")
        print(f"Transformers compatibility → evaluation_strategy: {self.SUPPORTS_EVAL_STRATEGY}, "
              f"callbacks: {self.SUPPORTS_CALLBACKS}, early_stop: {HAS_EARLY_STOP}")

    def train_mlm(
        self,
        model,
        tokenizer,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        mlm_probability: float = 0.15,
        max_length: int = 128,
        gradient_accumulation_steps: int = 1,
        fp16: bool = None,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        save_total_limit: int = 3,
        early_stopping: bool = True,
        early_stopping_patience: int = 3,
        seed: int = 42
    ):
        """
        Fine-tune a model using masked language modeling.
        
        This method implements the full training pipeline using HuggingFace Trainer when
        available, with automatic evaluation, early stopping, and checkpointing. If your
        transformers version doesn't support these features, it gracefully degrades to
        basic training without validation during training.
        
        The training uses a data collator that randomly masks 15% of tokens (by default)
        and trains the model to predict them. This is the standard MLM objective used in
        BERT and RoBERTa pretraining.
        
        Args:
            model: The model to fine-tune (should be a masked language model).
            tokenizer: Tokenizer corresponding to the model.
            train_texts: List of training texts.
            val_texts: Optional list of validation texts. If provided and your transformers
                      version supports it, enables validation during training and early stopping.
            learning_rate: Learning rate for the optimizer. 2e-5 is a good default for fine-tuning.
            num_epochs: Number of training epochs. Typically 2-5 epochs is sufficient.
            batch_size: Batch size per device. Actual batch size will be multiplied by
                       gradient_accumulation_steps.
            warmup_steps: Number of steps for learning rate warmup. Helps stabilize training.
            weight_decay: L2 regularization strength for AdamW optimizer.
            mlm_probability: Fraction of tokens to mask in each sequence. Standard is 0.15.
            max_length: Maximum sequence length for tokenization.
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating.
                                        Use this to simulate larger batch sizes.
            fp16: Whether to use mixed precision training. If None, automatically enabled on CUDA.
            save_steps: Save a checkpoint every N steps.
            eval_steps: Evaluate on validation set every N steps (if validation is available).
            logging_steps: Log training metrics every N steps.
            save_total_limit: Maximum number of checkpoints to keep. Older ones are deleted.
            early_stopping: Whether to use early stopping based on validation loss.
            early_stopping_patience: Number of evaluations without improvement before stopping.
            seed: Random seed for reproducibility.
        
        Returns:
            The fine-tuned model. Also saves checkpoints to output_dir.
        """
        print("\n" + "=" * 60)
        print("MASKED LANGUAGE MODEL TRAINING")
        print("=" * 60)

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Auto-detect whether to use FP16
        if fp16 is None:
            fp16 = torch.cuda.is_available()

        self._print_training_config(
            train_size=len(train_texts),
            val_size=len(val_texts) if val_texts else 0,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            mlm_probability=mlm_probability,
            fp16=fp16
        )

        # Prepare datasets
        print("\nPreparing datasets...")
        train_dataset = TextDataset(train_texts, tokenizer, max_length)
        val_dataset = TextDataset(val_texts, tokenizer, max_length) if val_texts else None

        print(f"Train dataset: {len(train_dataset)} examples")
        if val_dataset:
            print(f"Val dataset: {len(val_dataset)} examples")

        # Data collator handles random masking
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

        # Build TrainingArguments with compatibility checks
        common_kwargs = dict(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,

            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=1.0,

            fp16=fp16,

            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,

            # Only include these arguments if supported
            save_strategy="steps" if _supports_arg(TrainingArguments, "save_strategy") else "no",
            save_steps=save_steps if _supports_arg(TrainingArguments, "save_steps") else None,
            save_total_limit=save_total_limit if _supports_arg(TrainingArguments, "save_total_limit") else None,

            report_to="none" if _supports_arg(TrainingArguments, "report_to") else None,
            seed=seed,
            dataloader_num_workers=0,
        )

        if self.SUPPORTS_EVAL_STRATEGY and val_dataset is not None:
            # Modern configuration with evaluation during training
            common_kwargs.update(
                evaluation_strategy="steps",
                eval_steps=eval_steps,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False if self.SUPPORTS_GREATER_IS_BETTER else None
            )
        else:
            # Legacy configuration without evaluation_strategy
            if val_dataset is not None:
                print("\n[Compatibility] Your transformers version doesn't support 'evaluation_strategy'.")
                print("                Training will proceed without validation. Will evaluate at the end.")
            early_stopping = False  # Disable for compatibility

        # Create TrainingArguments, filtering out None values
        training_args = TrainingArguments(**{k: v for k, v in common_kwargs.items() if v is not None})

        # Build Trainer with optional callbacks
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Add early stopping callback if available
        callbacks = []
        if early_stopping and val_dataset and HAS_EARLY_STOP and self.SUPPORTS_CALLBACKS:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            trainer_kwargs["callbacks"] = callbacks

        print("\nStarting training...")
        start_time = time.time()

        # Some old versions don't accept 'callbacks' parameter
        try:
            trainer = Trainer(**trainer_kwargs)
        except TypeError:
            trainer_kwargs.pop("callbacks", None)
            trainer = Trainer(**trainer_kwargs)

        try:
            train_result = trainer.train()

            # Display training summary
            print("\n" + "=" * 60)
            print("TRAINING COMPLETE")
            print("=" * 60)

            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)

            # Final evaluation if validation set exists
            if val_dataset:
                print("\nRunning final evaluation...")
                eval_metrics = trainer.evaluate()
                trainer.log_metrics("eval", eval_metrics)

            elapsed_time = time.time() - start_time
            print(f"\nTotal training time: {elapsed_time/60:.1f} minutes")

            # Save the final model
            print(f"\nSaving final model to {self.output_dir / 'final'}...")
            final_path = self.output_dir / "final"
            trainer.save_model(str(final_path))
            tokenizer.save_pretrained(str(final_path))

            # Save training history
            history = {
                'train_loss': metrics.get('train_loss'),
                'eval_loss': eval_metrics.get('eval_loss') if val_dataset else None,
                'training_time': elapsed_time,
                'num_epochs': num_epochs,
            }

            with open(final_path / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

            print(f"Training completed successfully")
            return model

        except Exception as e:
            print(f"\nError during training: {e}")
            print("Attempting to save checkpoint before exiting...")

            try:
                emergency_path = self.output_dir / "emergency_checkpoint"
                model.save_pretrained(str(emergency_path))
                tokenizer.save_pretrained(str(emergency_path))
                print(f"Emergency checkpoint saved to {emergency_path}")
            except:
                print("Could not save emergency checkpoint")

            raise

    def train_simple(
        self,
        model,
        tokenizer,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        mlm_probability: float = 0.15,
        max_length: int = 128,
        save_every_epoch: bool = True
    ) -> Tuple:
        """
        Simple training loop without HuggingFace Trainer.
        
        This is a bare-bones training implementation that doesn't rely on the Trainer API.
        Use this if you want maximum control over the training process or if you're having
        compatibility issues with Trainer. It's also useful for understanding exactly what
        happens during training.
        
        The loop is straightforward: for each epoch, iterate through batches, compute loss,
        backpropagate, and update weights. Optionally evaluate on a validation set after
        each epoch and save checkpoints.
        
        Args:
            model: Model to train.
            tokenizer: Tokenizer for the model.
            train_texts: Training texts.
            val_texts: Optional validation texts.
            num_epochs: Number of epochs to train.
            batch_size: Batch size for training and validation.
            learning_rate: Learning rate for AdamW optimizer.
            mlm_probability: Fraction of tokens to mask.
            max_length: Maximum sequence length.
            save_every_epoch: Whether to save a checkpoint after each epoch.
        
        Returns:
            Tuple of (trained_model, training_history). History contains train and val loss
            per epoch.
        """
        print("\nSimple training mode (manual loop)")

        train_dataset = TextDataset(train_texts, tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = None
        val_loader = None
        if val_texts:
            val_dataset = TextDataset(val_texts, tokenizer, max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        history = {'train_loss': [], 'val_loss': []}

        print(f"\nTraining for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training phase
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc="Training"):
                # Prepare batch for masking
                bsz = batch['input_ids'].size(0)
                examples = [
                    {
                        'input_ids': batch['input_ids'][i],
                        'attention_mask': batch['attention_mask'][i]
                    } for i in range(bsz)
                ]
                masked = data_collator(examples)
                masked = {k: v.to(self.device) for k, v in masked.items()}

                # Forward pass
                outputs = model(**masked)
                loss = outputs.loss
                loss.backward()

                # Update weights
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            avg_train_loss = train_loss / max(1, len(train_loader))
            history['train_loss'].append(avg_train_loss)
            print(f"Train loss: {avg_train_loss:.4f}")

            # Validation phase
            if val_dataset:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        bsz = batch['input_ids'].size(0)
                        examples = [
                            {
                                'input_ids': batch['input_ids'][i],
                                'attention_mask': batch['attention_mask'][i]
                            } for i in range(bsz)
                        ]
                        masked = data_collator(examples)
                        masked = {k: v.to(self.device) for k, v in masked.items()}
                        outputs = model(**masked)
                        val_loss += outputs.loss.item()

                avg_val_loss = val_loss / max(1, len(val_loader))
                history['val_loss'].append(avg_val_loss)
                print(f"Val loss: {avg_val_loss:.4f}")

            # Save epoch checkpoint
            if save_every_epoch:
                checkpoint_path = self.output_dir / f"checkpoint-epoch-{epoch+1}"
                checkpoint_path.mkdir(exist_ok=True)
                model.save_pretrained(str(checkpoint_path))
                tokenizer.save_pretrained(str(checkpoint_path))
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_path = self.output_dir / "final"
        final_path.mkdir(exist_ok=True)
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        print("\nTraining completed")
        print(f"Final model saved to: {final_path}")

        return model, history

    def validate(
        self, model, tokenizer, val_texts: List[str],
        batch_size: int = 32, mlm_probability: float = 0.15,
        max_length: int = 128
    ) -> float:
        """
        Validate model on a validation set.
        
        Computes the average MLM loss on the validation set without updating model weights.
        This is useful for standalone evaluation or for checking model quality after training.
        
        Args:
            model: Model to validate.
            tokenizer: Tokenizer for the model.
            val_texts: Validation texts.
            batch_size: Batch size for validation.
            mlm_probability: Fraction of tokens to mask.
            max_length: Maximum sequence length.
        
        Returns:
            Average validation loss across all batches.
        """
        print("\nRunning validation...")

        val_dataset = TextDataset(val_texts, tokenizer, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                bsz = batch['input_ids'].size(0)
                examples = [
                    {
                        'input_ids': batch['input_ids'][i],
                        'attention_mask': batch['attention_mask'][i]
                    } for i in range(bsz)
                ]
                masked = data_collator(examples)
                masked = {k: v.to(self.device) for k, v in masked.items()}

                outputs = model(**masked)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def _print_training_config(
        self, train_size: int, val_size: int,
        learning_rate: float, num_epochs: int,
        batch_size: int, mlm_probability: float, fp16: bool
    ):
        """
        Display training configuration in a readable format.
        
        Args:
            train_size: Number of training examples.
            val_size: Number of validation examples.
            learning_rate: Learning rate being used.
            num_epochs: Number of training epochs.
            batch_size: Batch size per device.
            mlm_probability: Masking probability.
            fp16: Whether mixed precision is enabled.
        """
        print("\nTraining Configuration:")
        print(f"  Dataset:")
        print(f"    - Train examples: {train_size:,}")
        print(f"    - Val examples: {val_size:,}")
        print(f"  Hyperparameters:")
        print(f"    - Learning rate: {learning_rate}")
        print(f"    - Epochs: {num_epochs}")
        print(f"    - Batch size: {batch_size}")
        print(f"    - MLM probability: {mlm_probability}")
        print(f"  Optimization:")
        print(f"    - Mixed precision (FP16): {fp16}")
        print(f"    - Device: {self.device}")


# Standalone wrapper functions for convenience

def train_mlm(
    model, tokenizer, train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    output_dir: str = "./models/debiased", **kwargs
):
    """
    Train a model with MLM (convenience wrapper).
    
    Args:
        model: Model to train.
        tokenizer: Tokenizer for the model.
        train_texts: Training texts.
        val_texts: Optional validation texts.
        output_dir: Where to save the trained model.
        **kwargs: Additional arguments passed to MLMTrainer.train_mlm.
    
    Returns:
        The trained model.
    """
    trainer = MLMTrainer(output_dir=output_dir)
    return trainer.train_mlm(model, tokenizer, train_texts, val_texts, **kwargs)


def validate(model, tokenizer, val_texts: List[str], **kwargs) -> float:
    """
    Validate a model (convenience wrapper).
    
    Args:
        model: Model to validate.
        tokenizer: Tokenizer for the model.
        val_texts: Validation texts.
        **kwargs: Additional arguments passed to MLMTrainer.validate.
    
    Returns:
        Average validation loss.
    """
    trainer = MLMTrainer()
    return trainer.validate(model, tokenizer, val_texts, **kwargs)


if __name__ == "__main__":
    # Basic test to verify training works
    print("Testing training.py\n")

    from data_loader import DataLoader
    from model_manager import ModelManager

    print("=" * 60)
    print("TEST: MLM Fine-tuning")
    print("=" * 60)

    # Load debiasing corpus
    print("\n1. Loading debiasing corpus...")
    loader = DataLoader()
    corpus = loader.load_debiasing_corpus(balance=True)

    # Use small subset for quick testing
    train_texts = corpus['train'][:500]
    val_texts = corpus['val'][:100]

    print(f"Train: {len(train_texts)} examples")
    print(f"Val: {len(val_texts)} examples")

    # Load model
    print("\n2. Loading RoBERTa-base...")
    manager = ModelManager()
    model, tokenizer = manager.load_pretrained_roberta()

    # Train
    print("\n3. Fine-tuning model...")
    trainer = MLMTrainer(output_dir="./models/test_debiased")

    trained_model = trainer.train_mlm(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        num_epochs=1,
        batch_size=8,
        save_steps=100,
        eval_steps=100,
        logging_steps=50
    )

    print("\nTraining test completed")