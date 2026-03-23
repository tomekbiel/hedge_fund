import json
from datetime import datetime

# Read the notebook
with open('C:/python/hedge_fund/notebooks/data_loading_polars_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the SAVE PROCESSED DATA section
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'SAVE PROCESSED DATA' in ''.join(cell['source']):
        # Add timestamp code
        timestamp_code = '''    # Define paths with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_out_path = processed_dir / f"train_processed_v2_{timestamp}.parquet"
    test_out_path = processed_dir / f"test_processed_v2_{timestamp}.parquet"'''
        
        # Replace the old path definition
        old_code = '''    # Define paths
    train_out_path = processed_dir / "train_processed_v2.parquet"
    test_out_path = processed_dir / "test_processed_v2.parquet"'''
        
        # Join source lines
        source_text = ''.join(cell['source'])
        if old_code in source_text:
            new_source = source_text.replace(old_code, timestamp_code)
            cell['source'] = new_source
            print('Updated save paths with timestamp')
            break

# Save the notebook
with open('C:/python/hedge_fund/notebooks/data_loading_polars_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Notebook updated successfully')
