import polars as pl
from minio import Minio
import io
import json
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import os
import logging
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinioHandler:
    def __init__(self):
        load_dotenv()
        self.client = Minio(
            os.getenv('MINIO_ENDPOINT'),
            access_key=os.getenv('MINIO_ACCESS_KEY'),
            secret_key=os.getenv('MINIO_SECRET_KEY'),
            secure=False
        )
    
    def ensure_bucket_exists(self, bucket_name: str) -> None:
        try:
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"Creating bucket: {bucket_name}")
                self.client.make_bucket(bucket_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create bucket {bucket_name}: {e}")

class ChunkProcessor:
    def __init__(self):
        self.schema = pa.schema([
            ('document_name', pa.string()),
            ('arxiv_url_link', pa.string()),
            ('year', pa.int32()),
            ('category', pa.string()),
            ('total_chunks', pa.int32()),
            ('processed_chunks', pa.int32()),
            ('chunk_index', pa.int32()),
            ('text', pa.string()),
            ('title', pa.string()),
            ('abstract', pa.string()),
            ('summary', pa.string()),
            ('main_points', pa.string()),
            ('technical_terms', pa.string()),
            ('relationships', pa.string()),
            ('processing_time', pa.float64()),
            ('processing_attempt', pa.int32()),
            ('processing_timestamp', pa.string())
        ])
    
    def process_document(self, data: Dict) -> List[Dict]:
        """Convert document chunks into records"""
        records = []
        
        # Extract document-level metadata
        doc_metadata = {
            'document_name': str(data.get('document_name', '')),
            'arxiv_url_link': str(data.get('arxiv_url_link', '')),
            'year': int(data.get('year', 0)),
            'category': str(data.get('category', '')),
            'total_chunks': int(data.get('total_chunks', 0)),
            'processed_chunks': int(data.get('processed_chunks', 0))
        }
        
        # Process each chunk
        for chunk in data.get('chunks', []):
            record = {
                # Document metadata
                **doc_metadata,
                
                # Chunk data
                'chunk_index': int(chunk.get('chunk_index', 0)),
                'text': str(chunk.get('text', '')),
                'title': str(chunk.get('metadata', {}).get('title', '')),
                'abstract': str(chunk.get('metadata', {}).get('abstract', '')),
                'summary': str(chunk.get('analysis', {}).get('summary', '')),
                'main_points': json.dumps(chunk.get('analysis', {}).get('main_points', [])),
                'technical_terms': json.dumps(chunk.get('analysis', {}).get('technical_terms', [])),
                'relationships': json.dumps(chunk.get('analysis', {}).get('relationships', [])),
                'processing_time': float(chunk.get('processing_metadata', {}).get('processing_time', 0)),
                'processing_attempt': int(chunk.get('processing_metadata', {}).get('attempt', 0)),
                'processing_timestamp': datetime.now().isoformat()
            }
            records.append(record)
        
        return records

class ParquetConverter:
    def __init__(self):
        self.minio = MinioHandler()
        self.processor = ChunkProcessor()
    
    def _save_batch(self, records: List[Dict], dest_bucket: str, batch_num: int) -> str:
        """Save batch as Parquet with explicit schema"""
        # Convert records to Arrow Table with schema
        table = pa.Table.from_pylist(records, schema=self.processor.schema)
        
        # Write to buffer
        buffer = io.BytesIO()
        pq.write_table(
            table, 
            buffer,
            compression='snappy',
            version='2.6'  # Use older version for better compatibility
        )
        
        # Generate filename and upload
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parquet_path = f'processed/arxiv_batch_{batch_num}_{timestamp}.parquet'
        
        buffer.seek(0)
        self.minio.client.put_object(
            dest_bucket,
            parquet_path,
            data=buffer,
            length=len(buffer.getvalue())
        )
        
        logger.info(f"✓ Saved batch of {len(records)} records to {parquet_path}")
        return parquet_path
    
    def convert(self, source_bucket: str, dest_bucket: str, 
                batch_size: int = 100) -> List[str]:
        """Convert JSON files to Parquet"""
        self.minio.ensure_bucket_exists(dest_bucket)
        
        # List files
        files = list(self.minio.client.list_objects(source_bucket))
        if not files:
            raise ValueError(f"No files found in {source_bucket}")
        
        logger.info(f"Found {len(files)} files to process")
        
        created_files = []
        current_batch = []
        processed_count = 0
        
        for obj in files:
            try:
                # Read and process document
                doc_data = json.loads(
                    self.minio.client.get_object(
                        source_bucket, 
                        obj.object_name
                    ).read()
                )
                records = self.processor.process_document(doc_data)
                current_batch.extend(records)
                processed_count += 1
                
                # Save batch if reached size
                if len(current_batch) >= batch_size:
                    parquet_path = self._save_batch(
                        current_batch, 
                        dest_bucket, 
                        processed_count
                    )
                    created_files.append(parquet_path)
                    current_batch = []
                
                # Log progress
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(files)} files...")
                    
            except Exception as e:
                logger.error(f"Error processing {obj.object_name}: {e}")
                continue
        
        # Save final batch
        if current_batch:
            parquet_path = self._save_batch(
                current_batch, 
                dest_bucket, 
                processed_count
            )
            created_files.append(parquet_path)
        
        return created_files

def main():
    converter = ParquetConverter()
    
    try:
        parquet_files = converter.convert(
            source_bucket='arxiv-processed',
            dest_bucket='warehouse',
            batch_size=1000
        )
        
        logger.info("\nNext steps:")
        logger.info(f"Created {len(parquet_files)} Parquet files")
        logger.info("\nCreate Dremio table with:")
        logger.info("""
        CREATE TABLE nessie.arxiv_chunks AS 
        SELECT * FROM table(
            dfs.`@warehouse/processed/*.parquet`
            (type => 'parquet')
        );
        """)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("No files were modified")

if __name__ == "__main__":
    main()