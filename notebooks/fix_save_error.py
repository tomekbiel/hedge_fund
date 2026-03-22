import json

# Read notebook
with open('C:/python/hedge_fund/notebooks/data_loading_polars_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find SAVE PROCESSED DATA section and fix variable names
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'SAVE PROCESSED DATA' in ''.join(cell['source']):
        # Fix variable names from train_processed to train_full
        source_text = ''.join(cell['source'])
        
        # Replace train_processed with train_full
        fixed_source = source_text.replace('train_processed.write_parquet(train_out_path)', 'train_full.write_parquet(train_out_path)')
        fixed_source = fixed_source.replace('test_processed.write_parquet(test_out_path)', 'test_full.write_parquet(test_out_path)')
        fixed_source = fixed_source.replace('train_processed.shape[1]}', 'train_full.shape[1]}')
        
        cell['source'] = fixed_source
        print('Fixed variable names in save section')
        break

# Save notebook
with open('C:/python/hedge_fund/notebooks/data_loading_polars_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Notebook save error fixed!')
