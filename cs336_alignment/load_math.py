import pandas as pd
import os
import json
from datetime import datetime


parquet_file_path = 'data/MATH/math.parquet'

def create_train_val_split_from_parquet(parquet_path: str, output_dir: str, validation_split: float = 0.2):
    """Create both training and validation JSONL files from the main parquet dataset."""
    try:
        # Load the parquet file
        df = pd.read_parquet(parquet_path)
        print(f"Loaded dataset with {len(df)} examples")
        
        # Shuffle the dataset with a fixed random seed for reproducibility
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Split into training and validation
        validation_size = int(len(df) * validation_split)
        train_size = len(df) - validation_size
        
        validation_df = df_shuffled[:validation_size]
        train_df = df_shuffled[validation_size:]
        
        print(f"Split: {train_size} training examples, {validation_size} validation examples")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training set
        train_path = os.path.join(output_dir, 'train.jsonl')
        with open(train_path, 'w', encoding='utf-8') as f:
            for _, row in train_df.iterrows():
                json_obj = row.to_dict()
                f.write(json.dumps(json_obj) + '\n')
        
        # Save validation set  
        val_path = os.path.join(output_dir, 'validation.jsonl')
        with open(val_path, 'w', encoding='utf-8') as f:
            for _, row in validation_df.iterrows():
                json_obj = row.to_dict()
                f.write(json.dumps(json_obj) + '\n')
        
        print(f"Created training set with {len(train_df)} examples at {train_path}")
        print(f"Created validation set with {len(validation_df)} examples at {val_path}")
        
        # Save split metadata for reproducibility
        metadata = {
            'total_examples': len(df),
            'train_examples': len(train_df),
            'validation_examples': len(validation_df),
            'validation_split': validation_split,
            'random_seed': 42,
            'split_timestamp': datetime.now().isoformat(),
            'train_indices': train_df.index.tolist(),
            'validation_indices': validation_df.index.tolist()
        }
        
        metadata_path = os.path.join(output_dir, 'split_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved split metadata to {metadata_path}")
        
        return val_path, train_path
        
    except Exception as e:
        print(f"Error creating train/validation split: {e}")
        return None, None
    
if __name__ == "__main__":
    create_train_val_split_from_parquet(parquet_file_path, 'data/MATH')