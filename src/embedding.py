from typing import List, Dict, Union, Optional
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.logs.logger import setup_logger
from src.config.config import config

logger = setup_logger(__name__)

class EmbeddingClient:
    """Client for interacting with NVIDIA NeMo Embedding API."""
    
    def __init__(
        self,
        max_retries: int = 2,  # Reduced from 3
        retry_delay: int = 0.5  # Reduced from 1
    ):
        """Initialize embedding client with configuration and retry mechanism."""
        embedding_config = config._yaml_settings.get('embedding', {})
        self.url = embedding_config.get('endpoint')
        self.model_name = embedding_config.get('default_model')
        self.dimension = embedding_config.get('dimension')
        self.max_batch_size = embedding_config.get('max_batch_size', 8000)
        self.timeout = 10  # Reduced timeout
        self.truncate = "START"
        
        if not all([self.url, self.model_name, self.dimension]):
            raise ValueError("Missing required embedding configuration")
            
        # Optimize session setup with connection pooling
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Pre-configure headers
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"EmbeddingClient initialized with endpoint: {self.url}")
        logger.info(f"Using model: {self.model_name}")

    def get_embedding(
        self, 
        text: str,
        input_type: str = "passage",
        retry_attempts: int = 2
    ) -> Optional[List[float]]:
        """Get embedding for a single text input."""
        if not text:
            return None

        # Prepare request data
        payload = {
            "input": [text.strip()],
            "model": self.model_name,
            "input_type": input_type,
            "truncate": self.truncate
        }

        try:
            # Make request with optimized session
            response = self.session.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result.get("data", [])
                if embeddings:
                    embedding = embeddings[0].get("embedding", [])
                    if embedding and len(embedding) == self.dimension:
                        logger.info(f"Successfully generated embedding of dimension {self.dimension}")
                        return embedding
            
            logger.error(f"Failed to get embedding. Status: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Embedding request failed: {str(e)}")
            return None

    def validate_embedding_dimension(self, embedding: List[float]) -> bool:
        """Validate the dimension of returned embedding."""
        return isinstance(embedding, list) and len(embedding) == self.dimension

# Create a global instance
try:
    embedding_client = EmbeddingClient()
    logger.info("Global embedding client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize global embedding client: {str(e)}")
    embedding_client = None