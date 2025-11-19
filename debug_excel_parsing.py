#!/usr/bin/env python3
"""
Debug Excel file parsing for ground truth upload
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from ui.dashboard.components.ground_truth.file_handler import normalize_columns

def debug_excel_parsing():
    """Debug Excel file parsing issues"""
    print("🔍 Debugging Excel file parsing...")
    print("=" * 60)

    # Test with sample Vietnamese Excel-like data
    print("📄 Testing with sample Vietnamese Excel data...")

    # Simulate what might be in the Excel file
    sample_data = {
        'Câu hỏi': [
            'Machine learning là gì?',
            'Deep learning khác gì với machine learning?',
            'CNN được sử dụng để làm gì?'
        ],
        'Câu trả lời': [
            'Machine learning là một nhánh của trí tuệ nhân tạo cho phép máy tính học từ dữ liệu.',
            'Deep learning là một subset của machine learning sử dụng neural networks với nhiều layers.',
            'CNN được sử dụng chủ yếu cho computer vision tasks như image recognition.'
        ],
        'Nguồn': [
            'AI Basics',
            'Neural Networks',
            'Computer Vision'
        ]
    }

    df = pd.DataFrame(sample_data)
    print(f"✅ Created sample data with {len(df)} rows")
    print(f"Original columns: {list(df.columns)}")

    # Test normalization
    print("🔄 Testing column normalization...")
    normalized = normalize_columns(df)
    print(f"Normalized columns: {list(normalized.columns)}")
    print(f"Normalized shape: {normalized.shape}")

    # Show sample data
    print("\n📋 Sample normalized data:")
    for i, row in normalized.head(2).iterrows():
        print(f"  Q{i+1}: {row['question'][:50]}...")
        print(f"  A{i+1}: {row['answer'][:50]}...")
        print(f"  S{i+1}: {row['source']}")
        print()

    # Test edge cases
    print("🧪 Testing edge cases...")

    # Test with English columns
    english_data = {
        'Question': ['What is AI?', 'What is ML?'],
        'Answer': ['AI is artificial intelligence', 'ML is machine learning'],
        'Source': ['AI101', 'ML101']
    }
    df_english = pd.DataFrame(english_data)
    normalized_english = normalize_columns(df_english)
    print(f"✅ English columns: {list(df_english.columns)} → {list(normalized_english.columns)}")

    # Test with mixed case
    mixed_data = {
        'QUESTION': ['Q1?', 'Q2?'],
        'ANSWER': ['A1', 'A2'],
        'SOURCE': ['S1', 'S2']
    }
    df_mixed = pd.DataFrame(mixed_data)
    normalized_mixed = normalize_columns(df_mixed)
    print(f"✅ Mixed case columns: {list(df_mixed.columns)} → {list(normalized_mixed.columns)}")

    # Test with missing columns
    incomplete_data = {
        'Câu hỏi': ['Question 1?', 'Question 2?'],
        'Some Other Column': ['Data1', 'Data2']
    }
    df_incomplete = pd.DataFrame(incomplete_data)
    normalized_incomplete = normalize_columns(df_incomplete)
    print(f"✅ Incomplete columns: {list(df_incomplete.columns)} → {list(normalized_incomplete.columns)}")
    print(f"   Missing columns filled with empty strings: {normalized_incomplete.iloc[0]['answer'] == ''}")

    print("=" * 60)
    print("🔧 Column Normalization Rules:")
    print("Expected mappings (case-insensitive):")
    print("  question ← ['question', 'câu hỏi', 'cau hoi', 'q', 'query']")
    print("  answer ← ['answer', 'câu trả lời', 'cau tra loi', 'a', 'response']")
    print("  source ← ['source', 'nguồn', 'nguon', 's', 'reference']")
    print()
    print("💡 If your Excel file shows '0 parsed rows', check:")
    print("   1. Column names match one of the expected patterns above")
    print("   2. File is not corrupted or password-protected")
    print("   3. First row contains headers (not data)")
    print("   4. Try saving as CSV and re-uploading")

def test_excel_file_reading():
    """Test reading actual Excel file if available"""
    print("\n📂 Testing Excel file reading...")

    # Try to read the uploaded file if it exists
    excel_path = "5cau.xlsx"
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path)
            print(f"✅ Successfully read {excel_path}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print("   First few rows:")
            print(df.head(3))

            # Test normalization
            normalized = normalize_columns(df)
            print(f"   Normalized shape: {normalized.shape}")
            print(f"   Normalized columns: {list(normalized.columns)}")

        except Exception as e:
            print(f"❌ Failed to read {excel_path}: {e}")
    else:
        print(f"ℹ️  {excel_path} not found in current directory")
        print("   Upload the file through the dashboard to test")

def main():
    """Run debug tests"""
    print("🐛 EXCEL FILE PARSING DEBUG")
    print("=" * 60)

    try:
        debug_excel_parsing()
        test_excel_file_reading()

        print("=" * 60)
        print("🎯 DEBUG COMPLETE")
        print()
        print("If you're still getting 0 parsed rows:")
        print("1. Check your Excel column names")
        print("2. Ensure first row has headers")
        print("3. Try saving as CSV format")
        print("4. Share the column names for specific help")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())