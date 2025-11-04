"""
data_loader.py - Dataset management and preprocessing

This module handles downloading, loading, and preprocessing various bias evaluation
datasets including StereoSet, WinoBias, and CrowS-Pairs. It provides a unified interface
for managing all the data needed for bias detection and debiasing experiments.
"""

import os
import json
import pickle
import requests
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import random
from tqdm import tqdm


class DataLoader:
    """
    Manages all datasets used in the bias evaluation pipeline.
    
    This class handles the complete data workflow: downloading raw datasets from various
    sources, preprocessing them into a consistent format, and creating training corpora
    for debiasing experiments. It organizes data into raw and processed directories and
    provides convenient methods for accessing different dataset splits.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data loader with directory structure.
        
        Args:
            data_dir: Root directory for all data storage. Will create subdirectories
                     for raw and processed data if they don't exist.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directory structure if it doesn't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    # StereoSet dataset methods
    
    def download_stereoset(self) -> None:
        """
        Download StereoSet dataset files from GitHub.
        
        Fetches both dev and test splits of the StereoSet dataset. The test split
        may not always be available through the direct GitHub URL, in which case
        the method suggests alternative approaches like using the dev set or
        downloading from HuggingFace.
        """
        print("Downloading StereoSet...")
        
        stereoset_dir = self.raw_dir / "stereoset"
        stereoset_dir.mkdir(exist_ok=True)
        
        # URLs for the different splits
        files = {
            "dev.json": "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json",
            "test.json": "https://raw.githubusercontent.com/McGill-NLP/bias-bench/main/data/stereoset/test.json"
        }
        
        for filename, url in files.items():
            filepath = stereoset_dir / filename
            if filepath.exists():
                print(f"  {filename} already exists")
                continue
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"  Downloaded {filename}")
            except Exception as e:
                print(f"  Error downloading {filename}: {e}")
                
                # Provide helpful guidance if test split fails
                if filename == "test.json":
                    print(f"     test.json not available via direct download")
                    print(f"     You can use dev.json for testing, or download from HuggingFace")
                    print(f"     Alternative: Use 'dev.json' as test set (common practice)")
    
    def download_stereoset_huggingface(self) -> None:
        """
        Download StereoSet via HuggingFace datasets library.
        
        This is a more reliable fallback method that uses the HuggingFace datasets
        library to fetch StereoSet. It automatically converts the HuggingFace format
        to match the original StereoSet JSON structure for consistency with the rest
        of the pipeline.
        
        Requires the 'datasets' library to be installed.
        """
        try:
            from datasets import load_dataset
            
            print("Downloading StereoSet from HuggingFace...")
            
            stereoset_dir = self.raw_dir / "stereoset"
            stereoset_dir.mkdir(exist_ok=True)
            
            # Load the intrasentence split which is what we actually use
            dataset = load_dataset("McGill-NLP/stereoset", "intrasentence")
            
            for split_name in dataset.keys():
                data_split = dataset[split_name]
                
                # Convert to original StereoSet format for compatibility
                output_data = {"data": {"intrasentence": []}}
                
                for example in data_split:
                    processed = {
                        "id": example["id"],
                        "bias_type": example["bias_type"],
                        "context": example["context"],
                        "target": example.get("target", ""),
                        "sentences": []
                    }
                    
                    # Process sentences and convert numeric labels to text
                    for sent, label_list in zip(example["sentences"]["sentence"], 
                                                example["sentences"]["labels"]):
                        label_map = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}
                        label = label_map.get(label_list[0], "unrelated")
                        
                        processed["sentences"].append({
                            "sentence": sent,
                            "gold_label": label,
                            "id": f"{example['id']}_{len(processed['sentences'])}"
                        })
                    
                    output_data["data"]["intrasentence"].append(processed)
                
                # Map validation split to dev for consistency
                filename = "dev.json" if split_name == "validation" else f"{split_name}.json"
                filepath = stereoset_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
                
                print(f"  Downloaded {filename} ({len(output_data['data']['intrasentence'])} examples)")
            
        except ImportError:
            print("  HuggingFace datasets not installed. Install with: pip install datasets")
        except Exception as e:
            print(f"  Error with HuggingFace download: {e}")
    
    def load_stereoset(self, split: str = "test") -> Dict:
        """
        Load and preprocess StereoSet dataset.
        
        Args:
            split: Which split to load ('train', 'dev', or 'test'). Defaults to 'test'.
        
        Returns:
            Dictionary containing preprocessed examples organized by type (intrasentence
            and intersentence), with each example including context, bias type, and
            candidate sentences with labels.
        
        Raises:
            FileNotFoundError: If the dataset cannot be found and automatic download fails.
        """
        filepath = self.raw_dir / "stereoset" / f"{split}.json"
        
        # Try to download if file doesn't exist
        if not filepath.exists():
            print(f"StereoSet {split} not found.")
            
            if split == "test":
                print("Trying to download test.json...")
                self.download_stereoset()
                
                # Fall back to dev split if test is unavailable
                if not filepath.exists():
                    print(f"test.json not available. Using dev.json instead.")
                    print(f"(This is common practice - dev set works fine for evaluation)")
                    filepath = self.raw_dir / "stereoset" / "dev.json"
                    
                    # Try HuggingFace as last resort
                    if not filepath.exists():
                        print(f"Trying HuggingFace datasets...")
                        self.download_stereoset_huggingface()
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"StereoSet data not found. Please download manually or install 'datasets' library.\n"
                f"Run: pip install datasets\n"
                f"Then try again."
            )
        
        print(f"Loading StereoSet ({filepath.name})...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = self._preprocess_stereoset(data)
        
        # Print summary statistics
        print(f"Loaded {len(processed['intrasentence'])} intrasentence examples")
        print(f"  - Gender: {sum(1 for x in processed['intrasentence'] if x['bias_type'] == 'gender')}")
        print(f"  - Profession: {sum(1 for x in processed['intrasentence'] if x['bias_type'] == 'profession')}")
        print(f"  - Race: {sum(1 for x in processed['intrasentence'] if x['bias_type'] == 'race')}")
        
        return processed
    
    def _preprocess_stereoset(self, data: Dict) -> Dict:
        """
        Preprocess StereoSet into a clean, usable format.
        
        Takes the raw StereoSet JSON and converts it into a standardized format with
        consistent field names and structure. This makes it easier to work with in
        downstream evaluation tasks.
        
        Args:
            data: Raw StereoSet data loaded from JSON file.
        
        Returns:
            Dictionary with 'intrasentence' and 'intersentence' keys, each containing
            a list of examples with standardized fields.
        """
        processed = {
            'intrasentence': [],
            'intersentence': []
        }
        
        for example_type in ['intrasentence', 'intersentence']:
            if example_type not in data['data']:
                continue
            
            for example in data['data'][example_type]:
                processed_example = {
                    'id': example['id'],
                    'bias_type': example['bias_type'],
                    'context': example['context'],
                    'sentences': []
                }
                
                # Standardize sentence format
                for sentence in example['sentences']:
                    processed_example['sentences'].append({
                        'text': sentence['sentence'],
                        'label': sentence['gold_label'],  # stereotype, anti-stereotype, unrelated
                        'id': sentence['id']
                    })
                
                processed[example_type].append(processed_example)
        
        return processed
    
    # WinoBias dataset methods
    
    def download_winobias(self) -> None:
        """
        Download WinoBias dataset files from GitHub.
        
        Fetches all splits of the WinoBias dataset including both Type 1 and Type 2
        variants (which differ in the position of the target entity), and both
        pro-stereotyped and anti-stereotyped versions (which differ in whether the
        correct coreference follows or challenges gender stereotypes).
        """
        print("Downloading WinoBias...")
        
        winobias_dir = self.raw_dir / "winobias"
        winobias_dir.mkdir(exist_ok=True)
        
        base_url = "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data"
        
        # All the different file variants
        files = [
            "anti_stereotyped_type1.txt.dev",
            "anti_stereotyped_type1.txt.test", 
            "anti_stereotyped_type2.txt.dev",
            "anti_stereotyped_type2.txt.test",
            "pro_stereotyped_type1.txt.dev",
            "pro_stereotyped_type1.txt.test",
            "pro_stereotyped_type2.txt.dev",
            "pro_stereotyped_type2.txt.test"
        ]
        
        for filename in files:
            filepath = winobias_dir / filename
            if filepath.exists():
                print(f"  {filename} already exists")
                continue
            
            try:
                url = f"{base_url}/{filename}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"  Downloaded {filename}")
            except Exception as e:
                print(f"  Error downloading {filename}: {e}")
    
    def load_winobias(self, split: str = "test") -> List[Dict]:
        """
        Load and preprocess WinoBias dataset.
        
        WinoBias tests gender bias in coreference resolution. Each example contains
        a sentence with two entities and a pronoun that could refer to either one.
        The pro-stereotyped version has the pronoun refer to the stereotypical entity,
        while the anti-stereotyped version refers to the non-stereotypical one.
        
        Args:
            split: Which split to load ('dev' or 'test'). Defaults to 'test'.
        
        Returns:
            List of examples, each containing the text, entities, correct answer,
            and whether the example is pro-stereotyped or anti-stereotyped.
        
        Raises:
            FileNotFoundError: If WinoBias files are not found and download fails.
        """
        winobias_dir = self.raw_dir / "winobias"
        
        # Try to download if directory doesn't exist
        if not winobias_dir.exists():
            print("WinoBias not found. Attempting to download...")
            self.download_winobias()
        
        examples = []
        
        # Load all variants (Type 1 and 2, pro and anti)
        for stereotype_type in ['pro_stereotyped', 'anti_stereotyped']:
            for bias_type in ['type1', 'type2']:
                filename = f"{stereotype_type}_{bias_type}.txt.{split}"
                filepath = winobias_dir / filename
                
                if not filepath.exists():
                    print(f"Warning: {filename} not found")
                    continue
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        examples.append({
                            'text': line,
                            'label': stereotype_type,
                            'type': bias_type
                        })
        
        if not examples:
            raise FileNotFoundError(
                f"WinoBias files not found. Please download manually or check network connection."
            )
        
        print(f"Loaded {len(examples)} WinoBias examples ({split} split)")
        print(f"  - Pro-stereotyped: {sum(1 for x in examples if x['label'] == 'pro_stereotyped')}")
        print(f"  - Anti-stereotyped: {sum(1 for x in examples if x['label'] == 'anti_stereotyped')}")
        
        return examples
    
    # CrowS-Pairs dataset methods
    
    def download_crows_pairs(self) -> None:
        """
        Download CrowS-Pairs dataset from GitHub.
        
        CrowS-Pairs is a challenge dataset for measuring stereotypical biases in
        masked language models. Each example consists of a more stereotypical and
        a less stereotypical sentence that differ by just a few words.
        """
        print("Downloading CrowS-Pairs...")
        
        crows_dir = self.raw_dir / "crows_pairs"
        crows_dir.mkdir(exist_ok=True)
        
        url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
        filepath = crows_dir / "crows_pairs_anonymized.csv"
        
        if filepath.exists():
            print("  CrowS-Pairs already exists")
            return
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("  Downloaded crows_pairs_anonymized.csv")
        except Exception as e:
            print(f"  Error downloading CrowS-Pairs: {e}")
    
    def load_crows_pairs(self) -> pd.DataFrame:
        """
        Load and preprocess CrowS-Pairs dataset.
        
        Returns:
            DataFrame containing CrowS-Pairs examples with columns for the more
            stereotypical sentence (sent_more), less stereotypical sentence (sent_less),
            bias type, and other metadata.
        
        Raises:
            FileNotFoundError: If CrowS-Pairs file is not found and download fails.
        """
        filepath = self.raw_dir / "crows_pairs" / "crows_pairs_anonymized.csv"
        
        if not filepath.exists():
            print("CrowS-Pairs not found. Attempting to download...")
            self.download_crows_pairs()
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"CrowS-Pairs file not found. Please download manually."
            )
        
        print("Loading CrowS-Pairs...")
        df = pd.read_csv(filepath)
        
        print(f"Loaded {len(df)} CrowS-Pairs examples")
        print(f"Bias types: {df['bias_type'].value_counts().to_dict()}")
        
        return df
    
    # Corpus creation for debiasing
    
    def load_debiasing_corpus(self, balance: bool = True) -> Dict[str, List[str]]:
        """
        Create a combined corpus for debiasing from all available datasets.
        
        This method aggregates texts from WinoBias and CrowS-Pairs (both stereotyped
        and anti-stereotyped versions) to create training and validation sets for
        debiasing experiments. The resulting corpus can be used for continued pretraining
        or fine-tuning to reduce model bias.
        
        Args:
            balance: Whether to balance the number of stereotyped and anti-stereotyped
                    examples. Defaults to True.
        
        Returns:
            Dictionary with 'train' and 'val' keys, each containing a list of texts.
            Split is 90% train, 10% validation.
        
        Raises:
            ValueError: If no datasets are available for corpus creation.
        """
        print("Creating debiasing corpus...")
        texts = []
        
        # Add WinoBias examples
        try:
            winobias = self.load_winobias('dev')
            for example in winobias:
                texts.append({
                    'text': example['text'],
                    'source': 'winobias',
                    'label': example['label']
                })
            print(f"Added {len(winobias)} examples from WinoBias")
        except FileNotFoundError:
            print("WinoBias not available, skipping...")
        
        # Add CrowS-Pairs examples (both stereotyped and anti-stereotyped versions)
        try:
            crows = self.load_crows_pairs()
            for _, row in crows.iterrows():
                texts.append({
                    'text': row['sent_more'],
                    'source': 'crows_pairs',
                    'label': 'stereotyped'
                })
                texts.append({
                    'text': row['sent_less'],
                    'source': 'crows_pairs',
                    'label': 'anti_stereotyped'
                })
            print(f"Added {len(crows) * 2} examples from CrowS-Pairs")
        except FileNotFoundError:
            print("CrowS-Pairs not available, skipping...")
        
        if not texts:
            raise ValueError("No debiasing data available! Please download datasets first.")
        
        # Balance if requested
        if balance:
            texts = self.balance_dataset(texts)
        
        # Create train/val split (90/10)
        random.shuffle(texts)
        split_idx = int(0.9 * len(texts))
        
        corpus = {
            'train': [x['text'] for x in texts[:split_idx]],
            'val': [x['text'] for x in texts[split_idx:]]
        }
        
        print(f"\nDebiasing corpus ready:")
        print(f"  - Train: {len(corpus['train'])} examples")
        print(f"  - Val: {len(corpus['val'])} examples")
        print(f"  - Total: {len(texts)} examples")
        
        return corpus
    
    def balance_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Balance dataset between stereotyped and anti-stereotyped examples.
        
        This ensures that the model sees equal amounts of stereotypical and
        anti-stereotypical examples during training, which is important for
        effective debiasing.
        
        Args:
            data: List of examples, each containing a 'label' field indicating
                 whether it's stereotyped or anti-stereotyped.
        
        Returns:
            Balanced dataset with equal numbers of stereotyped and anti-stereotyped
            examples. If one class has more examples, it will be randomly downsampled.
        """
        print("  Balancing dataset...")
        
        # Separate into stereotyped and anti-stereotyped
        stereotyped = [x for x in data if 'pro' in x['label'] or x['label'] == 'stereotyped']
        anti_stereotyped = [x for x in data if 'anti' in x['label']]
        
        min_size = min(len(stereotyped), len(anti_stereotyped))
        
        # Sample equal amounts from each class
        balanced_data = (
            random.sample(stereotyped, min_size) +
            random.sample(anti_stereotyped, min_size)
        )
        
        print(f"    Before: {len(data)} examples")
        print(f"    After: {len(balanced_data)} examples ({min_size} per class)")
        
        return balanced_data
    
    # MLM dataset preparation
    
    def create_mlm_dataset(self, texts: List[str], save_path: Optional[str] = None) -> List[str]:
        """
        Prepare texts for Masked Language Modeling.
        
        Performs basic text cleaning and filtering to create a corpus suitable for
        MLM training. Note that the actual masking is done later by HuggingFace's
        DataCollatorForLanguageModeling during training.
        
        Args:
            texts: List of raw text strings to process.
            save_path: Optional path to save the cleaned texts. If provided, texts
                      are written one per line to this file.
        
        Returns:
            List of cleaned texts ready for MLM training.
        """
        print(f"Preparing MLM dataset with {len(texts)} texts...")
        
        # Basic cleaning and filtering
        cleaned_texts = []
        for text in tqdm(texts, desc="Cleaning texts"):
            # Normalize whitespace
            text = ' '.join(text.split())
            # Filter out texts that are too short or too long
            if 10 < len(text.split()) < 512:
                cleaned_texts.append(text)
        
        print(f"Cleaned dataset: {len(cleaned_texts)} texts")
        
        # Save if path is provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                for text in cleaned_texts:
                    f.write(text + '\n')
            print(f"Saved to {save_path}")
        
        return cleaned_texts
    
    # Utility methods
    
    def download_all(self) -> None:
        """
        Download all datasets used in the project.
        
        This is a convenience method that calls all individual download methods
        to fetch StereoSet, WinoBias, and CrowS-Pairs in one go.
        """
        print("=" * 50)
        print("DOWNLOADING ALL DATASETS")
        print("=" * 50 + "\n")
        
        self.download_stereoset()
        print()
        self.download_winobias()
        print()
        self.download_crows_pairs()
        
        print("\n" + "=" * 50)
        print("ALL DOWNLOADS COMPLETE")
        print("=" * 50)
    
    def save_processed(self, data: Dict, filename: str) -> None:
        """
        Save preprocessed data to disk using pickle.
        
        This is useful for caching preprocessed datasets to avoid reprocessing
        on subsequent runs.
        
        Args:
            data: Dictionary containing the processed data.
            filename: Name of the file to save (will be placed in processed_dir).
        """
        filepath = self.processed_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved processed data to {filepath}")
    
    def load_processed(self, filename: str) -> Dict:
        """
        Load preprocessed data from disk.
        
        Args:
            filename: Name of the file to load from processed_dir.
        
        Returns:
            Dictionary containing the loaded processed data.
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
        """
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Processed file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded processed data from {filepath}")
        return data
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics on all available datasets.
        
        Loads each dataset and computes summary statistics like total number of
        examples and breakdowns by category. Useful for understanding dataset
        composition and for verifying that downloads completed successfully.
        
        Returns:
            Dictionary with statistics for each dataset. Datasets that fail to load
            will have an 'error' key instead of statistics.
        """
        stats = {
            'stereoset': {},
            'winobias': {},
            'crows_pairs': {}
        }
        
        # StereoSet statistics
        try:
            stereoset = self.load_stereoset()
            stats['stereoset'] = {
                'total': len(stereoset['intrasentence']),
                'by_domain': {}
            }
            for domain in ['gender', 'profession', 'race']:
                count = sum(1 for x in stereoset['intrasentence'] if x['bias_type'] == domain)
                stats['stereoset']['by_domain'][domain] = count
        except:
            stats['stereoset'] = {'error': 'Not available'}
        
        # WinoBias statistics
        try:
            winobias = self.load_winobias()
            stats['winobias'] = {
                'total': len(winobias),
                'pro_stereotyped': sum(1 for x in winobias if x['label'] == 'pro_stereotyped'),
                'anti_stereotyped': sum(1 for x in winobias if x['label'] == 'anti_stereotyped')
            }
        except:
            stats['winobias'] = {'error': 'Not available'}
        
        # CrowS-Pairs statistics
        try:
            crows = self.load_crows_pairs()
            stats['crows_pairs'] = {
                'total': len(crows),
                'by_type': crows['bias_type'].value_counts().to_dict()
            }
        except:
            stats['crows_pairs'] = {'error': 'Not available'}
        
        return stats


# Standalone wrapper functions for backward compatibility
# These allow using the loader functions without instantiating the DataLoader class

def load_stereoset(split: str = "test", data_dir: str = "./data") -> Dict:
    """
    Load StereoSet dataset (convenience wrapper).
    
    Args:
        split: Which split to load ('train', 'dev', or 'test').
        data_dir: Root directory for data storage.
    
    Returns:
        Preprocessed StereoSet data.
    """
    loader = DataLoader(data_dir)
    return loader.load_stereoset(split)


def load_debiasing_corpus(data_dir: str = "./data", balance: bool = True) -> Dict[str, List[str]]:
    """
    Create debiasing corpus (convenience wrapper).
    
    Args:
        data_dir: Root directory for data storage.
        balance: Whether to balance stereotyped and anti-stereotyped examples.
    
    Returns:
        Dictionary with 'train' and 'val' text lists.
    """
    loader = DataLoader(data_dir)
    return loader.load_debiasing_corpus(balance)


def create_mlm_dataset(texts: List[str], save_path: Optional[str] = None) -> List[str]:
    """
    Prepare MLM dataset (convenience wrapper).
    
    Args:
        texts: List of raw texts to process.
        save_path: Optional path to save cleaned texts.
    
    Returns:
        List of cleaned texts.
    """
    loader = DataLoader()
    return loader.create_mlm_dataset(texts, save_path)


def balance_dataset(data: List[Dict]) -> List[Dict]:
    """
    Balance dataset by class (convenience wrapper).
    
    Args:
        data: List of examples with 'label' field.
    
    Returns:
        Balanced dataset with equal class sizes.
    """
    loader = DataLoader()
    return loader.balance_dataset(data)


if __name__ == "__main__":
    # Basic testing and demonstration of the data loader
    print("Testing data_loader.py\n")
    
    loader = DataLoader()
    
    # Download all datasets
    loader.download_all()
    
    print("\n" + "=" * 50)
    print("TESTING LOADING FUNCTIONS")
    print("=" * 50 + "\n")
    
    # Test loading different datasets
    stereoset = loader.load_stereoset('test')
    debiasing_corpus = loader.load_debiasing_corpus()
    
    # Display statistics
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50 + "\n")
    stats = loader.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\nAll tests passed")