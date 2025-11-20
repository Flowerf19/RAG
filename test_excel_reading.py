import pandas as pd
import unicodedata

# Test reading the Excel file
file_path = r"c:\Users\ENGUYEHWC\Downloads\5cau.xlsx"  # Update this to the actual path

try:
    # Try to read Excel with more options
    df = pd.read_excel(file_path, engine='openpyxl', sheet_name=0, header=0)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"All rows:")
    print(df)
    
    # Check for non-null values
    print(f"\nNon-null counts:\n{df.count()}")
    
    # Check data types
    print(f"\nData types:\n{df.dtypes}")

    # Test normalization
    df_columns = [col.lower().strip() for col in df.columns]
    df_columns_normalized = [unicodedata.normalize('NFC', col) for col in df_columns]
    print(f"Normalized: {df_columns_normalized}")

    # Test mapping
    column_mappings = {
        'question': ['question', 'câu hỏi', 'cau hoi', 'q', 'query', 'câu hoi', 'cau hỏi', 'câu hỏi (question)'],
        'answer': ['answer', 'câu trả lời', 'cau tra loi', 'a', 'response', 'cau tra loi', 'cau trả lời', 'câu trả lời (answer)'],
        'source': ['source', 'nguồn', 'nguon', 's', 'reference', 'nguon', 'nguồn', 'nguồn (source)']
    }

    for target_col, possible_names in column_mappings.items():
        possible_names_normalized = [unicodedata.normalize('NFC', name) for name in possible_names]
        print(f"Looking for {target_col}: {possible_names_normalized}")

        for possible_name in possible_names_normalized:
            if possible_name in df_columns_normalized:
                col_idx = df_columns_normalized.index(possible_name)
                print(f"  Found match: '{df.columns[col_idx]}' -> {target_col}")
                break
        else:
            print(f"  No match found for {target_col}")

except Exception as e:
    print(f"Error: {e}")