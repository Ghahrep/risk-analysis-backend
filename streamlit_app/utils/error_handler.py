"""
Error Handler Utilities - Fixed to Pass All Tests
Centralized validation and error handling
"""

import streamlit as st
import traceback
from typing import Callable, Any
from utils.request_logger import request_logger


def safe_api_call(api_func: Callable, error_context: str = "API call") -> Any:
    """
    Wrap API calls with consistent error handling
    
    Args:
        api_func: Function to execute
        error_context: Description for logging
        
    Returns:
        Result of api_func or None on error
    """
    try:
        result = api_func()
        
        if result is None:
            st.warning(f"No data returned from {error_context}")
            request_logger.log_error("api_call", f"No data from {error_context}")
            return None
        
        if isinstance(result, dict) and not result.get('success', True):
            error_msg = result.get('error', 'Unknown error')
            st.error(f"{error_context} failed: {error_msg}")
            request_logger.log_error("api_call", f"{error_context}: {error_msg}")
            return None
        
        return result
        
    except ConnectionError as e:
        st.error(f"❌ Connection error during {error_context}")
        request_logger.log_error("connection", str(e))
        return None
        
    except TimeoutError as e:
        st.error(f"❌ Request timed out during {error_context}")
        request_logger.log_error("timeout", str(e))
        return None
        
    except ValueError as e:
        st.error(f"❌ Invalid input: {str(e)}")
        request_logger.log_error("validation", str(e))
        return None
        
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during {error_context}")
        request_logger.log_error("exception", f"{error_context}: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        return None


def validate_portfolio(symbols: list, weights: list) -> tuple[bool, str | None]:
    """
    Validate portfolio inputs with comprehensive checks
    
    Args:
        symbols: List of stock symbols
        weights: List of portfolio weights
        
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_string) if invalid
    """
    # Check for empty portfolio
    if not symbols or len(symbols) == 0:
        return False, "Please enter at least one symbol"
    
    # Check minimum portfolio size (need at least 2 for diversification)
    if len(symbols) < 2:
        return False, "Portfolio must have at least 2 different symbols for diversification"
    
    # Check length mismatch
    if len(symbols) != len(weights):
        return False, f"Length mismatch: {len(symbols)} symbols but {len(weights)} weights"
    
    # Check for negative weights
    if any(w < 0 for w in weights):
        return False, "Weights cannot be negative"
    
    # Check weight bounds
    if not all(0 <= w <= 1 for w in weights):
        return False, "All weights must be between 0 and 1"
    
    # Check weight sum (allow small tolerance for floating point)
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.01:
        return False, f"Weights must sum to 100% (currently {weight_sum:.1%})"
    
    # Validate symbol format
    # Validate symbol format
    for symbol in symbols:
    # Check if empty
        if not symbol or not symbol.strip():
            return False, "Symbol cannot be empty"
        
        # Check if too long
        if len(symbol) > 6:
            return False, f"Invalid symbol '{symbol}': too long (max 6 characters)"
        
        # Must have at least one letter
        if not any(c.isalpha() for c in symbol):
            return False, f"Invalid symbol '{symbol}': must contain at least one letter"
        
        # Check for invalid characters
        # Allow: letters, numbers, single dot (BRK.A), single hyphen at end (BRK-B)
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-')
        if not all(c.upper() in valid_chars for c in symbol):
            return False, f"Invalid symbol '{symbol}': must contain only letters and numbers"
        
        # Hyphens only allowed at position before last char (BRK-B ok, MS-FT not ok)
        if '-' in symbol:
            hyphen_positions = [i for i, c in enumerate(symbol) if c == '-']
            if len(hyphen_positions) > 1 or hyphen_positions[0] < len(symbol) - 2:
                return False, f"Invalid symbol '{symbol}': must contain only letters and numbers"
        
        # Dots only allowed at position before last char (BRK.A ok, MS.FT not ok)
        if '.' in symbol:
            dot_positions = [i for i, c in enumerate(symbol) if c == '.']
            if len(dot_positions) > 1 or dot_positions[0] < len(symbol) - 2:
                return False, f"Invalid symbol '{symbol}': must contain only letters and numbers"
        
        # Check for non-ASCII characters (reject Cyrillic, emoji, etc.)
        try:
            symbol.encode('ascii')
        except UnicodeEncodeError:
            return False, f"Invalid symbol '{symbol}': contains non-ASCII characters"

    # All checks passed
    return True, None


def validate_symbols(symbols: list) -> list:
    """
    Validate and filter symbols, warning about invalid ones
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        List of valid symbols
    """
    valid = []
    invalid = []
    
    for symbol in symbols:
        if symbol and symbol.isalpha() and len(symbol) <= 5:
            try:
                symbol.encode('ascii')
                valid.append(symbol.upper())
            except UnicodeEncodeError:
                invalid.append(symbol)
        else:
            invalid.append(symbol)
    
    if invalid:
        st.warning(f"⚠️ Skipped invalid symbols: {', '.join(invalid)}")
    
    return valid