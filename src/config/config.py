# src/config/config.py
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class combining .env and YAML settings with validation."""
    
    def __init__(self):
        self._yaml_settings = self._load_yaml()
        self._load_env_vars()
        self._setup_paths()
        
    def _load_yaml(self):
        """Load settings from config.yaml and handle missing file gracefully."""
        yaml_path = Path(__file__).parent / "config.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_env_vars(self):
        """Load and validate required environment variables."""
        
        # MinIO settings
        self.MINIO_ENDPOINT = f"http://{os.getenv('MINIO_ENDPOINT')}"
        self.MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
        self.MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
        self.MINIO_BUCKET_NAME = self._yaml_settings.get('minio', {}).get('bucket_name', "default-bucket")

        # Milvus settings
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self._yaml_settings.get('milvus', {}).get('host'))
        self.MILVUS_PORT = int(os.getenv('MILVUS_PORT', self._yaml_settings.get('milvus', {}).get('port', 19530)))
        self.MILVUS_USER = os.getenv('MILVUS_USER')
        self.MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD')
        self.MILVUS_TLS_ENABLED = os.getenv('MILVUS_TLS_ENABLED', 'false').lower() == 'true'
        self.MILVUS_COLLECTION = self._yaml_settings.get('milvus', {}).get('collection_name', "default_collection")

        # Embedding and Language Model settings
        embedding_model = self._yaml_settings.get('embedding_model', {})
        self.EMBEDDING_MODEL_URL = embedding_model.get('url', "http://localhost:8000/v1/embeddings")
        self.EMBEDDING_MODEL_NAME = embedding_model.get('model_name', "default_embedding_model")
        
        lm_service = self._yaml_settings.get('lm_service', {})
        self.LM_SERVICE_URL = lm_service.get('url', "http://localhost:8000/v1/chat/completions")
        self.LM_MODEL_NAME = lm_service.get('model_name', "default_language_model")

        # Logging settings
        logging_settings = self._yaml_settings.get('logging', {})
        self.LOG_LEVEL = logging_settings.get('level', 'INFO')
        self.LOG_FORMAT = logging_settings.get('format', '%(asctime)s - %(levelname)s - %(message)s')

    def _setup_paths(self):
        """Set up data paths and ensure directories exist."""
        paths_config = self._yaml_settings.get('paths', {})
        self.PATHS = {
            'raw_pdfs': Path(paths_config.get('raw_pdfs', 'data/raw_pdfs')),
            'raw_text': Path(paths_config.get('raw_text', 'data/raw_text')),
            'structured_text': Path(paths_config.get('structured_text', 'data/structured_text')),
            'embeddings': Path(paths_config.get('embeddings', 'data/embeddings'))
        }
        
        # Ensure directories exist
        for path_name, path in self.PATHS.items():
            path.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate that all required settings are present."""
        required_vars = {
            'MINIO_ENDPOINT': self.MINIO_ENDPOINT,
            'MINIO_ACCESS_KEY': self.MINIO_ACCESS_KEY,
            'MINIO_SECRET_KEY': self.MINIO_SECRET_KEY,
            'MILVUS_HOST': self.MILVUS_HOST,
            'MILVUS_USER': self.MILVUS_USER,
            'MILVUS_PASSWORD': self.MILVUS_PASSWORD,
        }
        
        missing_vars = [name for name, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True

# Create global configuration instance
config = Config()

# Validate configuration upon loading
config.validate()