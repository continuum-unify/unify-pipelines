#!/usr/bin/env python3
"""
Test script for Aleph Manifold Pipeline
This script tests both streaming and non-streaming responses from the pipeline.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Generator, Iterator, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our pipeline
# Assuming the file is named 'aleph_manifold_pipeline.py'
try:
    from aleph_vision_pipeline_2 import Pipeline
except ImportError:
    logger.error("Failed to import Pipeline. Make sure the file exists in the current directory.")
    sys.exit(1)

async def validate_api_key(pipeline: Pipeline) -> bool:
    """Simple validation of API key."""
    if not pipeline.valves.ANTHROPIC_API_KEY:
        logger.error("No API key provided! Set the ANTHROPIC_API_KEY environment variable.")
        return False
    
    # Check if API key starts with valid prefix
    if not pipeline.valves.ANTHROPIC_API_KEY.startswith(("sk-", "anthropic-")):
        logger.warning("API key has unusual format, may not be valid.")
    
    return True

async def process_non_streaming_response(response: Dict[str, Any]) -> None:
    """Handle and display a non-streaming response."""
    if "error" in response:
        logger.error(f"Error from API: {response['error']}")
        return

    print("\n🔹 Non-Streaming Response:")
    print("-" * 80)
    
    # Extract and print content
    try:
        if "content" in response:
            content_blocks = response["content"]
            for block in content_blocks:
                if block["type"] == "text":
                    print(block["text"])
        else:
            print(json.dumps(response, indent=2))
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        print(json.dumps(response, indent=2))
    
    print("-" * 80)

def process_streaming_response(response: Generator) -> None:
    """Handle and display a streaming response from Anthropic API."""
    print("\n🔹 Streaming Response:")
    print("-" * 80)
    
    try:
        response_text = ""
        
        # Process each chunk from the generator
        for chunk in response:
            # Check if chunk is a string
            if isinstance(chunk, str):
                # Check if it's an error message
                if chunk.startswith("API error:") or chunk.startswith("Streaming error:"):
                    logger.error(chunk)
                    print(f"\n{chunk}")
                else:
                    # It's regular text content - print it and accumulate
                    print(chunk, end="", flush=True)
                    response_text += chunk
        
        # Print a newline at the end for better formatting
        print()
        
        # Log the complete response for debugging
        if response_text:
            logger.info(f"Received complete streaming response ({len(response_text)} chars)")
                    
    except Exception as e:
        logger.error(f"Error processing streaming response: {e}")
    
    print("-" * 80)

async def run_test() -> None:
    """Run the pipeline test."""
    
    # Initialize the pipeline
    pipeline = Pipeline()
    
    # Make sure startup is called
    await pipeline.on_startup()
    
    # Validate API key
    if not await validate_api_key(pipeline):
        return
    
    # User input for test message
    user_message = input("Enter your test message: ").strip()
    
    # Ensure message is not empty
    if not user_message:
        logger.error("Test message cannot be empty.")
        return
    
    # Display available models
    print("\nAvailable models:")
    for model_id, model_name in pipeline.model_mapping.items():
        print(f"  {model_id}: {model_name}")
    
    # Model selection
    model_id = input(f"\nEnter model ID: ").strip()
    
    # Validate model_id
    if model_id not in pipeline.model_mapping:
        logger.error(f"Invalid model ID. Available options: {list(pipeline.model_mapping.keys())}")
        return
    
    logger.info(f"Using model: {pipeline.model_mapping[model_id]}")
    
    # Ask if user wants to add an image
    add_image = input("\nAdd an image? (yes/no): ").strip().lower() in ["y", "yes"]
    attachments = []
    
    if add_image:
        image_path = input("Enter image path or URL: ").strip()
        if os.path.exists(image_path):
            try:
                import base64
                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")
                    # Guess the MIME type based on file extension
                    if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
                        mime_type = "image/jpeg"
                    elif image_path.lower().endswith(".png"):
                        mime_type = "image/png"
                    else:
                        mime_type = "image/jpeg"  # Default
                    
                    attachments.append({
                        "type": mime_type,
                        "data": f"data:{mime_type};base64,{img_data}"
                    })
                    logger.info(f"Added image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
        else:
            logger.warning(f"Image file not found: {image_path}")
            
    # Define messages in the expected format
    messages = [{"role": "user", "content": user_message}]
    
    # Ask user if they want streaming enabled
    streaming_mode = input("\nEnable streaming? (yes/no): ").strip().lower() in ["y", "yes"]
    
    # Define request body
    body = {
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": streaming_mode,
        "attachments": attachments
    }
    
    # Create properly formatted request for inlet processing
    inlet_body = {
        "model": model_id,
        "messages": messages,
        "stream": streaming_mode,
        "attachments": attachments
    }
    
    try:
        # Process through inlet first (as would happen in OpenWebUI)
        processed_body = await pipeline.inlet(inlet_body)
        
        # Run the pipe method
        logger.info("Starting pipeline test...")
        response = pipeline.pipe(user_message, model_id, messages, body)
        
        # Handle response based on streaming mode
        if streaming_mode:
            process_streaming_response(response)
        else:
            await process_non_streaming_response(response)
            
        # Process through outlet (for non-streaming)
        if not streaming_mode and isinstance(response, dict):
            processed_response = await pipeline.outlet(response)
            logger.info("Response processed through outlet")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    # Ensure shutdown is called
    finally:
        await pipeline.on_shutdown()

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY environment variable is not set!")
        api_key = input("Enter your Anthropic API key: ").strip()
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            print("No API key provided. Exiting.")
            sys.exit(1)
    
    # Run the test
    asyncio.run(run_test())