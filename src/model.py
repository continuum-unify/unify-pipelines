# src/models/model.py

import json
from typing import Optional, Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config.config import config
from src.logs import setup_logger

logger = setup_logger(__name__)

# Module-level singleton interface
_model_interface: Optional['ModelInterface'] = None

class ModelInterface:
    """Interface for interacting with LLM API with retry logic and error handling."""
    
    def __init__(self, config_instance=None):
        """Initialize model interface with configuration."""
        self.config = config_instance or config
        self.setup_constants()
        self.session = self.create_requests_session()

    def setup_constants(self) -> None:
        """Set up constants from configuration."""
        try:
            settings = self.config._yaml_settings
            self.ENDPOINT_URL = settings.get('endpoint', {}).get('url') or settings.get('embedding', {}).get('endpoint')
            self.MODEL_NAME = settings.get('model', {}).get('name') or settings.get('embedding', {}).get('default_model')
            self.GENERATION_PARAMS = settings.get('generation', {})
            
            error_config = settings.get('error_handling', {})
            self.API_RETRY_LIMIT = error_config.get('max_retries', 3)
            self.BACKOFF_FACTOR = error_config.get('backoff_factor', 0.5)
            self.RETRY_STATUS_CODES = tuple(error_config.get('retry_status_codes', [500, 502, 504]))
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Set defaults if config fails
            self.ENDPOINT_URL = "http://192.168.13.50:32484/v1/chat/completions"
            self.MODEL_NAME = "meta/llama3-8b-instruct"
            self.GENERATION_PARAMS = {
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 1,
                "n": 1,
                "stream": False,
                "frequency_penalty": 0.0
            }
            self.API_RETRY_LIMIT = 3
            self.BACKOFF_FACTOR = 0.5
            self.RETRY_STATUS_CODES = (500, 502, 504)

    def create_requests_session(self) -> requests.Session:
        """Create and configure requests session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=self.API_RETRY_LIMIT,
            read=self.API_RETRY_LIMIT,
            connect=self.API_RETRY_LIMIT,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=self.RETRY_STATUS_CODES,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def interact_with_model(
        self, 
        messages: List[Dict[str, str]], 
        json_schema: Optional[Dict] = None, 
        max_tokens: Optional[int] = None, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Interact with the model API with retry handling."""
        if not messages:
            logger.error("Cannot interact with model: messages is empty")
            return None

        data = self._prepare_request_data(messages, json_schema, max_tokens, **kwargs)
        
        try:
            response = self.session.post(
                self.ENDPOINT_URL,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json=data,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error during model interaction: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Error during model interaction: {e}")

        return None

    def _prepare_request_data(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict],
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare request data with all parameters."""
        data = {
            "model": self.MODEL_NAME,
            "messages": messages,
            "max_tokens": max_tokens or self.GENERATION_PARAMS.get('max_tokens', 2000),
            "temperature": kwargs.get('temperature', self.GENERATION_PARAMS.get('temperature', 0.3)),
            "top_p": kwargs.get('top_p', self.GENERATION_PARAMS.get('top_p', 1)),
            "n": kwargs.get('n', self.GENERATION_PARAMS.get('n', 1)),
            "stream": kwargs.get('stream', self.GENERATION_PARAMS.get('stream', False)),
            "frequency_penalty": kwargs.get('frequency_penalty', self.GENERATION_PARAMS.get('frequency_penalty', 0.0)),
        }

        if json_schema:
            data["nvext"] = {"guided_json": json_schema}
            
        # Add any additional nvext parameters
        for param in ['guided_choice', 'guided_regex', 'guided_grammar']:
            if param in kwargs and kwargs[param] is not None:
                if "nvext" not in data:
                    data["nvext"] = {}
                data["nvext"][param] = kwargs[param]

        return data

def get_model_interface(config_instance=None) -> ModelInterface:
    """Get or create singleton ModelInterface instance."""
    global _model_interface
    if _model_interface is None:
        _model_interface = ModelInterface(config_instance)
    return _model_interface

def interact_with_model(*args, **kwargs) -> Optional[Dict[str, Any]]:
    """Convenience function for model interaction."""
    return get_model_interface().interact_with_model(*args, **kwargs)