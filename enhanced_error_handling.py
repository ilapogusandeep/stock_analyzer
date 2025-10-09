"""Enhanced error handling and user feedback utilities."""
from typing import Any, Dict, Optional, Tuple
import streamlit as st
import time
from functools import wraps

class AnalysisError(Exception):
    """Base class for analysis errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

def with_error_handling(section_name: str):
    """Decorator for handling errors in analysis sections."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with st.spinner(f"Loading {section_name}..."):
                    return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {section_name}: {str(e)}")
                with st.expander("Error Details"):
                    st.write(f"Error Type: {type(e).__name__}")
                    st.write(f"Error Message: {str(e)}")
                    if hasattr(e, 'details'):
                        st.write("Additional Details:", e.details)
                return None
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    raise last_error
        return wrapper
    return decorator

def show_loading_progress(total_steps: int):
    """Context manager for showing loading progress."""
    progress = st.progress(0)
    step = 0
    
    class ProgressContext:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            progress.empty()
        
        def update(self, message: str):
            nonlocal step
            step += 1
            progress.progress(step / total_steps)
            st.write(message)
    
    return ProgressContext()

def handle_api_error(func):
    """Decorator for handling API-specific errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = type(e).__name__
            if 'ConnectionError' in error_type:
                st.error("⚠️ Network Error: Please check your internet connection")
            elif 'Timeout' in error_type:
                st.error("⚠️ Request Timeout: The server took too long to respond")
            elif 'ValueError' in error_type:
                st.error("⚠️ Invalid Input: Please check your input values")
            else:
                st.error(f"⚠️ An error occurred: {str(e)}")
            return None
    return wrapper