# src/prepare_network_data.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import json
from config_manager import ConfigManager

def main():
    """
    Automates Step 2 of the pipeline: creating the author-topic network file.
    """
    print("--- Running Step 2: Preparing Author-Topic Network Data ---")
    config = ConfigManager()

    # Get input paths from config
    cleaned_snapshot_path = config.get_path('cleaned_snapshot', section='static_inputs')
    topic_metadata_path = config.get_path('latest_topic_metadata_path')

    # Load data
    arxiv_df = pd.read_csv(cleaned_snapshot_path)
    
    with open(topic_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Find the document topics file from the metadata
    topic_dir = Path(topic_metadata_path).parent
    doc_topics_filename = metadata['file_references']['document_topics']
    doc_topics_path = topic_dir / doc_topics_filename
    doc_topics_df = pd.read_csv(doc_topics_path)
    
    # Crucially, exclude outlier topic -1
    doc_topics_df = doc_topics_df[doc_topics_df['topic'] != -1]
    
    # Join the two dataframes
    merged_df = pd.merge(doc_topics_df[['id', 'topic']], arxiv_df[['id', 'authors_parsed']], on='id')
    
    # Select and reorder columns
    final_df = merged_df[['topic', 'id', 'authors_parsed']]
    
    # Get output path from config and save
    output_path = config.get_path('author_topic_network_path')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"Successfully created author-topic network file at: {output_path}")
    print(f"Total entries (papers in topics): {len(final_df)}")

if __name__ == "__main__":
    main()