#!/usr/bin/env python3
"""
Debug token reading from various sources
"""

import os
print('=== Debug Token Reading ===')

# Check secrets.toml path
secrets_path = os.path.join(os.getcwd(), '.streamlit', 'secrets.toml')
print(f'Secrets path: {secrets_path}')
print(f'Exists: {os.path.exists(secrets_path)}')

if os.path.exists(secrets_path):
    try:
        import tomllib
        with open(secrets_path, 'rb') as f:
            secrets = tomllib.load(f)
            token = secrets.get('HF_TOKEN')
            masked_token = "***" + token[-10:] if token else "None"
            print(f'Token from secrets.toml: {masked_token}')
    except ImportError:
        print('tomllib not available, trying tomli...')
        try:
            import tomli
            with open(secrets_path, 'rb') as f:
                secrets = tomli.load(f)
                token = secrets.get('HF_TOKEN')
                masked_token = "***" + token[-10:] if token else "None"
                print(f'Token from secrets.toml: {masked_token}')
        except ImportError:
            print('tomli not available either')
    except Exception as e:
        print(f'Error reading secrets: {e}')

# Check env vars
env_token = os.getenv('HF_TOKEN')
masked_env = "***" + env_token[-10:] if env_token else "None"
print(f'Env HF_TOKEN: {masked_env}')

# Test HfFolder
try:
    from huggingface_hub import HfFolder
    hf_token = HfFolder.get_token()
    masked_hf = "***" + hf_token[-10:] if hf_token else "None"
    print(f'HfFolder token: {masked_hf}')
except Exception as e:
    print(f'HfFolder error: {e}')