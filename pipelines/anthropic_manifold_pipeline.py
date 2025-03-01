"""
title: Anthropic Manifold Pipeline
author: Your Name
date: 2024-01-20
version: 1.0
license: MIT
description: A pipeline for generating text using the Anthropic API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import sseclient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        """Initialize the Anthropic pipeline."""
        self.type = "manifold"
        self.id = "aleph"  # Changed to match our naming scheme
        self.name = "aleph"  # Changed to match our naming scheme
        
        # Define model mapping
        self.model_mapping = {
            "1": "claude-3-5-sonnet-20241022",
            "2": "claude-3-sonnet-20240229",
            "3": "claude-3-opus-20240229",
            "4": "claude-3-haiku-20240307"
        }
        
        # Initialize valves with API key from environment
        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")}
        )
        self.url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()
        logger.info("Aleph pipeline initialized")

    def update_headers(self):
        """Update request headers with current API key."""
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY
        }
        logger.debug("Headers updated")

    def get_available_models(self):
        """Return list of available models with custom names."""
        return [
            {"id": model_id, "name": f"Aleph {model_id}"} 
            for model_id in self.model_mapping.keys()
        ]

    async def on_startup(self):
        """Handle startup tasks."""
        logger.info(f"Starting {self.name} pipeline")
        if not self.valves.ANTHROPIC_API_KEY:
            logger.warning("No Anthropic API key provided")

    async def on_shutdown(self):
        """Handle shutdown tasks."""
        logger.info(f"Shutting down {self.name} pipeline")

    async def on_valves_updated(self):
        """Handle valve updates."""
        self.update_headers()
        logger.info("Valves updated")

    def pipelines(self) -> List[dict]:
        """Return available pipelines (models)."""
        return self.get_available_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Process a message through the Anthropic API."""
        try:
            # Map our custom model name to Anthropic model ID
            anthropic_model_id = self.model_mapping.get(model_id)
            if not anthropic_model_id:
                raise ValueError(f"Unknown model: {model_id}")
                
            # Clean up body
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            # Process messages
            processed_messages = []
            for message in messages:
                if isinstance(message.get("content"), str):
                    processed_messages.append({
                        "role": message["role"],
                        "content": [{"type": "text", "text": message["content"]}]
                    })

            # Prepare payload with the actual Anthropic model ID
            payload = {
                "model": anthropic_model_id,  # Use mapped Anthropic model ID
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.7),
                "stream": body.get("stream", False)
            }

            logger.info(f"Processing request with model: Aleph {model_id} (Anthropic: {anthropic_model_id})")
            
            # Handle streaming vs non-streaming
            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return self.get_completion(payload)

        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}")
            return f"Error: {str(e)}"

    def stream_response(self, payload: dict) -> Generator:
        """Handle streaming responses from Anthropic API."""
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()

            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    if data["type"] == "content_block_start":
                        yield data["content_block"]["text"]
                    elif data["type"] == "content_block_delta":
                        yield data["delta"]["text"]
                    elif data["type"] == "message_stop":
                        break
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    logger.error(f"Unexpected data structure: {e}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming error: {str(e)}")
            raise Exception(f"Streaming error: {str(e)}")

    def get_completion(self, payload: dict) -> str:
        """Handle non-streaming responses from Anthropic API."""
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"] if "content" in result and result["content"] else ""
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Completion error: {str(e)}")
            raise Exception(f"Completion error: {str(e)}")
