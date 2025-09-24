import os
import json
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """
    Load configuration based on APP_ENV environment variable.
    Defaults to 'local' if APP_ENV is not set.
    
    Returns:
        dict: Configuration dictionary containing all settings from the JSON file
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    env = os.getenv('APP_ENV', 'local')
    
    file_name = os.path.join('configs/', f'{env}.json')
    
    try:
        with open(file_name, 'r') as f:
            config_data = json.load(f)
            return config_data
            
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to read config file ({file_name}): {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse config file ({file_name}): {str(e)}")