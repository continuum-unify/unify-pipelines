from pymilvus import connections, Collection, utility, MilvusClient
import json
from src.logs.logger import logger

def inspect_milvus():
    """Inspect Milvus databases and collections."""
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host="10.106.175.99",
            port="19530"
        )
        logger.info("Successfully connected to Milvus")
        
        # Create client for admin operations
        client = MilvusClient(uri="http://10.106.175.99:19530")
        
        # List databases
        databases = client.list_databases()
        logger.info("\n=== Databases ===")
        for db in databases:
            logger.info(f"  • {db}")
            
            # Connect to specific database
            connections.connect(
                alias=db,
                host="10.106.175.99",
                port="19530",
                db_name=db
            )
            
            # List collections in this database
            collections = utility.list_collections()
            if collections:
                logger.info(f"\n  Collections in '{db}':")
                for coll_name in collections:
                    logger.info(f"    ◦ {coll_name}")
                    
                    # Get collection details
                    collection = Collection(coll_name)
                    schema = collection.schema
                    
                    logger.info(f"\n    Collection '{coll_name}' Schema:")
                    logger.info("    Fields:")
                    for field in schema.fields:
                        logger.info(f"      - {field.name}")
                        logger.info(f"        Type: {field.dtype}")
                        if hasattr(field, "dim"):
                            logger.info(f"        Dimension: {field.dim}")
                        if hasattr(field, "max_length"):
                            logger.info(f"        Max Length: {field.max_length}")
                        if field.description:
                            logger.info(f"        Description: {field.description}")
                    
                    # Get collection statistics
                    logger.info(f"\n    Statistics:")
                    logger.info(f"      Row count: {collection.num_entities}")
                    
                    # Get index information
                    indexes = collection.indexes
                    if indexes:
                        logger.info(f"\n    Indexes:")
                        for idx in indexes:
                            logger.info(f"      - Field: {idx.field_name}")
                            logger.info(f"        Type: {idx.params.get('index_type')}")
                            logger.info(f"        Params: {idx.params.get('params', {})}")
                    
            logger.info("\n" + "="*50)

    except Exception as e:
        logger.error(f"Inspection failed: {str(e)}")
    finally:
        # Cleanup connections
        for db in databases:
            if connections.has_connection(db):
                connections.disconnect(db)
        if connections.has_connection("default"):
            connections.disconnect("default")
        logger.info("\nInspection completed")

if __name__ == "__main__":
    logger.info("Starting Milvus Inspection")
    inspect_milvus()