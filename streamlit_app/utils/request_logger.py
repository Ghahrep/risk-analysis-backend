"""
Request logging for monitoring and debugging
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Setup logging directory
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger("streamlit_app")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_DIR / "streamlit_requests.log")
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class RequestLogger:
    """Log API requests and responses"""
    
    @staticmethod
    def log_request(endpoint: str, params: Dict[str, Any], user_id: str = "anonymous"):
        """Log outgoing API request"""
        logger.info(f"REQUEST | User: {user_id} | Endpoint: {endpoint} | Params: {json.dumps(params)}")
    
    @staticmethod
    def log_response(endpoint: str, success: bool, duration: float, error: str = None):
        """Log API response"""
        status = "SUCCESS" if success else "FAILED"
        log_msg = f"RESPONSE | Endpoint: {endpoint} | Status: {status} | Duration: {duration:.2f}s"
        
        if error:
            log_msg += f" | Error: {error}"
            logger.error(log_msg)
        else:
            logger.info(log_msg)
    
    @staticmethod
    def log_user_action(action: str, details: Dict[str, Any] = None):
        """Log user interactions"""
        log_msg = f"USER_ACTION | Action: {action}"
        if details:
            log_msg += f" | Details: {json.dumps(details)}"
        logger.info(log_msg)
    
    # ADD THIS METHOD
    @staticmethod
    def log_error(error_type: str, message: str):
        """Log errors from error_handler"""
        logger.error(f"ERROR | Type: {error_type} | Message: {message}")


# Create singleton instance
request_logger = RequestLogger()