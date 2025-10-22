from pathlib import Path
import os

print('Current working directory:', os.getcwd())
print('Script file location:', __file__)

# Calculate secrets path as in HuggingFaceEmbedder
script_path = Path(__file__)
secrets_path = script_path.parent.parent.parent / '.streamlit' / 'secrets.toml'
print('Calculated secrets path:', secrets_path)
print('Secrets path exists:', secrets_path.exists())
print('Absolute path:', secrets_path.absolute())

# Test reading
if secrets_path.exists():
    try:
        import tomllib
        with open(secrets_path, 'rb') as f:
            secrets = tomllib.load(f)
            print('Secrets loaded')
            if 'huggingface' in secrets:
                token = secrets['huggingface'].get('api_token')
                print('Token found:', bool(token))
    except Exception as e:
        print('Error:', e)
