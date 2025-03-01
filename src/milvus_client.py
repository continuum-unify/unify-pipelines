from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, MilvusClient as PyMilvusClient
from src.logs.logger import logger  # Updated import path
from src.config.config import config  # Updated import path
import time

class MilvusClient:
    def __init__(self, config=config):
        """Initialize MilvusClient with configuration."""
        self.alias = "default"
        self.config = config
        self.collection_config = self.config._yaml_settings['collections']['documents']
        self.collection_name = self.collection_config['name']
        self.collection = None
        self.database = "arxiv"
        self.pymilvus_client = None
        self._define_schema()
        logger.info(f"Initialized MilvusClient with collection name: {self.collection_name} for database: {self.database}")

    def _define_schema(self):
        """Define the collection schema based on configuration."""
        try:
            self.fields = []
            for field_config in self.collection_config['fields']:
                field_params = {
                    'name': field_config['name'],
                    'dtype': getattr(DataType, field_config['dtype']),
                    'description': field_config.get('description', '')
                }
                
                # Add field-specific parameters
                for param in ['is_primary', 'auto_id']:
                    if field_config.get(param):
                        field_params[param] = True
                
                for param in ['max_length', 'dim']:
                    if param in field_config:
                        field_params[param] = field_config[param]
                
                self.fields.append(FieldSchema(**field_params))

            self.schema = CollectionSchema(
                fields=self.fields,
                description=self.collection_config['description'],
                enable_dynamic_field=self.collection_config['enable_dynamic_field']
            )
            logger.info("Schema defined successfully")
        except Exception as e:
            logger.error(f"Error defining schema: {str(e)}")
            raise

    def _init_pymilvus_client(self):
        """Initialize the PyMilvus client for database operations."""
        try:
            self.pymilvus_client = PyMilvusClient(
                uri=f"http://{self.config.MILVUS_HOST}:{self.config.MILVUS_PORT}",
                user=self.config.MILVUS_USER,
                password=self.config.MILVUS_PASSWORD
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PyMilvus client: {str(e)}")
            return False

    def setup_database(self):
        """Setup database and ensure it exists."""
        try:
            if not self._init_pymilvus_client():
                raise Exception("Failed to initialize PyMilvus client")

            databases = self.pymilvus_client.list_databases()
            logger.info(f"Available databases: {databases}")

            if self.database not in databases:
                logger.info(f"Creating database '{self.database}'")
                self.pymilvus_client.create_database(self.database)
                logger.info(f"Database '{self.database}' created successfully")
            else:
                logger.info(f"Database '{self.database}' already exists")
            return True
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            return False

    def connect(self, retries=3, delay=2):
        """Establish connection to Milvus with retry logic."""
        try:
            # Setup database first
            if not self.setup_database():
                raise Exception("Database setup failed")

            for attempt in range(retries):
                try:
                    logger.info(f"Connecting to Milvus at {self.config.MILVUS_HOST}:{self.config.MILVUS_PORT} (Database: {self.database})")
                    connections.connect(
                        alias=self.alias,
                        host=self.config.MILVUS_HOST,
                        port=self.config.MILVUS_PORT,
                        user=self.config.MILVUS_USER,
                        password=self.config.MILVUS_PASSWORD,
                        secure=self.config.MILVUS_TLS_ENABLED,
                        db_name=self.database
                    )
                    logger.info(f"Successfully connected to Milvus {self.database} database")
                    return True
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        return False
        except Exception as e:
            logger.error(f"Connection setup failed: {str(e)}")
            return False

    def create_collection(self):
        """Create collection with configured schema and indexes."""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                self._verify_indexes()
            else:
                logger.info(f"Creating new collection {self.collection_name}")
                self.collection = Collection(
                    name=self.collection_name,
                    schema=self.schema,
                    using=self.alias
                )
                logger.info(f"Collection {self.collection_name} created successfully")
                self._create_indexes()
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False

    def _create_field_index(self, field_config):
        """Create index for a specific field based on configuration."""
        try:
            if 'index' not in field_config:
                return True

            index_config = field_config['index']
            logger.info(f"Creating index for {field_config['name']} with config: {index_config}")
            
            self.collection.create_index(
                field_name=field_config['name'],
                index_params=index_config
            )
            logger.info(f"Created index for {field_config['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index for {field_config['name']}: {str(e)}")
            return False

    def _verify_indexes(self):
        """Verify existing indexes match configuration."""
        try:
            existing_indexes = {idx.field_name: idx for idx in self.collection.indexes}
            logger.info(f"Existing indexes: {list(existing_indexes.keys())}")
            
            for field in self.collection_config['fields']:
                if 'index' in field:
                    field_name = field['name']
                    if field_name not in existing_indexes:
                        logger.info(f"Creating missing index for {field_name}")
                        self._create_field_index(field)
            
            logger.info("All required indexes verified")
            return True
        except Exception as e:
            logger.error(f"Error verifying indexes: {str(e)}")
            return False

    def _create_indexes(self):
        """Create all configured indexes."""
        try:
            for field in self.collection_config['fields']:
                self._create_field_index(field)
            return True
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            return False

    def get_collection(self):
        """Get and load collection."""
        try:
            if not self.collection:
                if not utility.has_collection(self.collection_name):
                    return None
                self.collection = Collection(self.collection_name)
            
            if not self.collection.is_empty:
                self.collection.load()
                logger.info(f"Collection '{self.collection_name}' loaded for querying")
            
            return self.collection
        except Exception as e:
            logger.error(f"Error getting collection: {str(e)}")
            return None

    def load_collection(self):
        """Load collection into memory."""
        try:
            if self.collection and not self.collection.is_empty:
                self.collection.load()
                logger.info(f"Collection {self.collection_name} loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load collection: {str(e)}")
            return False

    def release_collection(self):
        """Release collection from memory."""
        try:
            if self.collection:
                self.collection.release()
                logger.info(f"Collection {self.collection_name} released from memory")
                self.collection = None
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to release collection: {str(e)}")
            return False

    def drop_collection(self):
        """Drop collection."""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.collection = None
                logger.info(f"Collection {self.collection_name} dropped")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to drop collection: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            if self.collection:
                self.release_collection()
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
                logger.info("Disconnected from Milvus")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect: {str(e)}")
            return False

# Create global instance
milvus_client = MilvusClient()