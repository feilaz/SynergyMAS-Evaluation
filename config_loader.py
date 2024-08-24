# config_loader.py

import os
import configparser
from pathlib import Path

def load_config():
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent / 'config.ini'
    config.read(config_path)
    return config

def get_config_value(config, section, key, env_var=None, default=None):
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
    try:
        value = config[section][key]
    except KeyError:
        if default is not None:
            return str(Path(__file__).parent / default)
        raise
    
    if section == 'PATHS':
        # Resolve relative paths
        return str(Path(__file__).parent / value)
    return value

# Define default paths
DEFAULT_PATHS = {
    'CHROMA_DB_DIR': 'chroma_db',
    'MRA_DATA_PATH': 'rag/MRA',
    'PD_DATA_PATH': 'rag/PD',
    'SM_DATA_PATH': 'rag/SM'
}

# Function to get all necessary config values
def get_all_config_values(config):
    return {
        'OPENAI_API_KEY': get_config_value(config, 'API_KEYS', 'OPENAI_API_KEY', 'OPENAI_API_KEY'),
        'LANGCHAIN_API_KEY': get_config_value(config, 'API_KEYS', 'LANGCHAIN_API_KEY', 'LANGCHAIN_API_KEY'),
        'LANGCHAIN_PROJECT': get_config_value(config, 'LANGCHAIN', 'LANGCHAIN_PROJECT'),
        'LANGCHAIN_TRACING_V2': config.getboolean('LANGCHAIN', 'LANGCHAIN_TRACING_V2', fallback=False),
        'OPENAI_MODEL': get_config_value(config, 'MODELS', 'OPENAI_MODEL'),
        'CHROMA_DB_DIR': get_config_value(config, 'PATHS', 'CHROMA_DB_DIR', default=DEFAULT_PATHS['CHROMA_DB_DIR']),
        'MRA_DATA_PATH': get_config_value(config, 'PATHS', 'MRA_DATA_PATH', default=DEFAULT_PATHS['MRA_DATA_PATH']),
        'PD_DATA_PATH': get_config_value(config, 'PATHS', 'PD_DATA_PATH', default=DEFAULT_PATHS['PD_DATA_PATH']),
        'SM_DATA_PATH': get_config_value(config, 'PATHS', 'SM_DATA_PATH', default=DEFAULT_PATHS['SM_DATA_PATH']),
        'NEO4J_URL': get_config_value(config, 'NEO4J', 'NEO4J_URL'),
        'NEO4J_USERNAME': get_config_value(config, 'NEO4J', 'NEO4J_USERNAME'),
        'NEO4J_PASSWORD': get_config_value(config, 'NEO4J', 'NEO4J_PASSWORD'),
    }