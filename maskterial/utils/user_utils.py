"""
User utilities for MaskTerial server
Handles user-specific data storage and retrieval
"""

import os
from pathlib import Path
from typing import Optional


def get_user_data_dir(base_dir: str, user_id: str) -> str:
    """
    Get or create user-specific data directory
    
    Args:
        base_dir: Base directory for all user data
        user_id: User identifier
        
    Returns:
        Path to user-specific data directory
    """
    user_dir = os.path.join(base_dir, "users", user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_user_model_dir(base_model_dir: str, user_id: str, model_type: str) -> str:
    """
    Get or create user-specific model directory
    
    Args:
        base_model_dir: Base directory for models
        user_id: User identifier
        model_type: Type of model (e.g., 'classification_models', 'segmentation_models')
        
    Returns:
        Path to user-specific model directory
    """
    user_model_dir = os.path.join(base_model_dir, model_type, "users", user_id)
    os.makedirs(user_model_dir, exist_ok=True)
    return user_model_dir


def get_user_upload_dir(base_dir: str, user_id: str) -> str:
    """
    Get or create user-specific upload directory
    
    Args:
        base_dir: Base directory for uploads
        user_id: User identifier
        
    Returns:
        Path to user-specific upload directory
    """
    upload_dir = os.path.join(base_dir, "uploads", user_id)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def get_user_training_dir(base_dir: str, user_id: str) -> str:
    """
    Get or create user-specific training directory
    
    Args:
        base_dir: Base directory for training data
        user_id: User identifier
        
    Returns:
        Path to user-specific training directory
    """
    training_dir = os.path.join(base_dir, "training", user_id)
    os.makedirs(training_dir, exist_ok=True)
    return training_dir


def sanitize_user_id(user_id: Optional[str]) -> str:
    """
    Sanitize user ID to prevent path traversal attacks
    
    Args:
        user_id: User identifier to sanitize
        
    Returns:
        Sanitized user ID
    """
    if not user_id:
        return "test_user"
    
    # Remove any path separators and potentially dangerous characters
    sanitized = user_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    
    # Only allow alphanumeric, underscore, and hyphen
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ["_", "-"])
    
    # If empty after sanitization, use default
    if not sanitized:
        return "test_user"
    
    return sanitized


def log_user_action(base_dir: str, user_id: str, action: str, details: dict = None):
    """
    Log user actions for audit purposes
    
    Args:
        base_dir: Base directory for logs
        user_id: User identifier
        action: Action description
        details: Additional details about the action
    """
    import json
    from datetime import datetime
    
    log_dir = os.path.join(base_dir, "logs", "user_actions")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{user_id}.jsonl")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "action": action,
        "details": details or {}
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

