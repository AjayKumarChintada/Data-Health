"""
JSON Utilities for safe JSON operations
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def safe_json_dump(data: Any, file_path: str, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Safely dump data to JSON file with error handling
    
    Args:
        data: Data to serialize
        file_path: Path to output file
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII output
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def safe_json_load(file_path: str) -> Any:
    """
    Safely load data from JSON file with error handling
    
    Args:
        file_path: Path to input file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def safe_json_serialize(data: Any) -> str:
    """
    Safely serialize data to JSON string with error handling
    
    Args:
        data: Data to serialize
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Error serializing data to JSON: {e}")
        raise

def safe_json_deserialize(json_string: str) -> Any:
    """
    Safely deserialize JSON string with error handling
    
    Args:
        json_string: JSON string to deserialize
        
    Returns:
        Deserialized data
    """
    try:
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"Error deserializing JSON string: {e}")
        raise
