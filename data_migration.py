#!/usr/bin/env python3
"""
Ground Truth Data Migration for Ragas
=====================================

This script migrates ground truth data from CSV format to Ragas-compatible format.

Current format (CSV):
- STT: Serial number
- Câu hỏi: Question (Vietnamese)
- Câu trả lời: Answer (Vietnamese)
- Nguồn: Source text

Ragas format (JSON/Dataset):
- question: User's question
- answer: Expected answer (ground truth)
- contexts: List of context chunks (from Nguồn field)
- ground_truth: Same as answer for Ragas compatibility
"""

import csv
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_ground_truth_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth data from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries with ground truth data
    """
    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'stt': row.get('STT', ''),
                'question': row.get('Câu hỏi', ''),
                'answer': row.get('Câu trả lời', ''),
                'source': row.get('Nguồn', '')
            })

    logger.info(f"Loaded {len(data)} ground truth samples from {csv_path}")
    return data


def convert_to_ragas_format(ground_truth_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert ground truth data to Ragas-compatible format.

    Args:
        ground_truth_data: List of ground truth dictionaries

    Returns:
        List of Ragas-formatted dictionaries
    """
    ragas_data = []

    for item in ground_truth_data:
        # Split source text into context chunks (simple splitting by double newlines)
        source_text = item['source']
        contexts = [chunk.strip() for chunk in source_text.split('\n\n') if chunk.strip()]

        # If no chunks, use the whole source as one context
        if not contexts:
            contexts = [source_text]

        ragas_item = {
            'question': item['question'],
            'answer': item['answer'],  # This will be the ground truth for evaluation
            'contexts': contexts,
            'ground_truth': item['answer']  # Ragas expects this field
        }

        ragas_data.append(ragas_item)

    logger.info(f"Converted {len(ragas_data)} samples to Ragas format")
    return ragas_data


def save_ragas_dataset(ragas_data: List[Dict[str, Any]], output_path: str):
    """
    Save Ragas-formatted data to JSON file.

    Args:
        ragas_data: List of Ragas-formatted dictionaries
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(ragas_data)} samples to {output_path}")


def create_huggingface_dataset(ragas_data: List[Dict[str, Any]], output_path: str):
    """
    Create a HuggingFace dataset from Ragas data.

    Args:
        ragas_data: List of Ragas-formatted dictionaries
        output_path: Path to save the dataset
    """
    try:
        from datasets import Dataset

        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(ragas_data)

        # Save as JSON (can be loaded later)
        dataset.to_json(output_path)
        logger.info(f"Saved HuggingFace dataset to {output_path}")

    except ImportError:
        logger.warning("datasets library not available, skipping HuggingFace dataset creation")


def migrate_ground_truth_data(
    input_csv: str = "test_ground_truth.csv",
    output_json: str = "ground_truth_ragas.json",
    output_hf: str = "ground_truth_ragas_dataset.json"
):
    """
    Complete migration pipeline from CSV to Ragas format.

    Args:
        input_csv: Input CSV file path
        output_json: Output JSON file path
        output_hf: Output HuggingFace dataset path
    """
    logger.info("Starting ground truth data migration...")

    # Load CSV data
    ground_truth_data = load_ground_truth_csv(input_csv)

    # Convert to Ragas format
    ragas_data = convert_to_ragas_format(ground_truth_data)

    # Save JSON format
    save_ragas_dataset(ragas_data, output_json)

    # Save HuggingFace dataset format
    create_huggingface_dataset(ragas_data, output_hf)

    logger.info("Migration completed successfully!")
    return ragas_data


def validate_ragas_format(ragas_data: List[Dict[str, Any]]) -> bool:
    """
    Validate that the data is in correct Ragas format.

    Args:
        ragas_data: List of Ragas-formatted dictionaries

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['question', 'answer', 'contexts', 'ground_truth']

    for i, item in enumerate(ragas_data):
        for field in required_fields:
            if field not in item:
                logger.error(f"Sample {i}: Missing required field '{field}'")
                return False

        if not isinstance(item['contexts'], list):
            logger.error(f"Sample {i}: 'contexts' must be a list")
            return False

        if not item['contexts']:
            logger.error(f"Sample {i}: 'contexts' cannot be empty")
            return False

    logger.info("Ragas format validation passed")
    return True


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run migration
    ragas_data = migrate_ground_truth_data()

    # Validate format
    if validate_ragas_format(ragas_data):
        print("✅ Migration successful! Data is ready for Ragas evaluation.")
        print(f"📊 Migrated {len(ragas_data)} samples")
        print("📁 Output files:")
        print("   - ground_truth_ragas.json (JSON format)")
        print("   - ground_truth_ragas_dataset.json (HuggingFace dataset)")
    else:
        print("❌ Migration failed! Please check the logs above.")
        exit(1)