"""
title: Aleph Manifold Pipeline
author: Your Name
date: 2024-01-20
version: 1.3
license: MIT
description: A pipeline for text generation and image analysis using the Anthropic API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
import json
from typing import List, Dict, Any
from pydantic import BaseModel
import sseclient
import logging

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        MAX_IMAGE_SIZE: int = 100 * 1024 * 1024  # 100MB limit
        MAX_IMAGES_PER_REQUEST: int = 20  # Anthropic's limit for claude.ai
        MAX_IMAGE_DIMENSION: int = 8000  # Maximum pixel dimension
        MIN_IMAGE_DIMENSION: int = 200   # Minimum recommended dimension

    def __init__(self):
        """Initialize the Aleph pipeline."""
        self.type = "aleph_vision"
        self.id = "aleph_vision"
        self.name = "aleph"
        
        # Define model mapping
        self.model_mapping = {
            "1": "claude-3-7-sonnet-20250219",
            "2": "claude-3-5-haiku-20241022",
            "3": "claude-3-5-sonnet-20241022",
            "4": "claude-3-5-sonnet-20240620",
            "5": "claude-3-opus-20240229",
            "6": "claude-3-sonnet-20240229",
            "7": "claude-3-haiku-20240307"
        }
        
        # Initialize valves with API key from environment
        self.valves = self.Valves(
            **{
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
                "MAX_IMAGE_SIZE": int(os.getenv("MAX_IMAGE_SIZE", 100 * 1024 * 1024)),
                "MAX_IMAGES_PER_REQUEST": int(os.getenv("MAX_IMAGES_PER_REQUEST", 20)),
                "MAX_IMAGE_DIMENSION": int(os.getenv("MAX_IMAGE_DIMENSION", 8000)),
                "MIN_IMAGE_DIMENSION": int(os.getenv("MIN_IMAGE_DIMENSION", 200))
            }
        )
        self.url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()
        logger.info("Aleph pipeline initialized")

    def update_headers(self):
        """Update request headers with current API key."""
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY.strip()
        }
        logger.debug("Headers updated")

    def validate_api_key(self) -> bool:
        """Validate the Anthropic API key."""
        if not self.valves.ANTHROPIC_API_KEY:
            logger.error("No Anthropic API key provided")
            return False
            
        if not self.valves.ANTHROPIC_API_KEY.startswith("sk-ant"):
            logger.error("Invalid Anthropic API key format")
            return False
        
        try:
            response = requests.get(
                "https://api.anthropic.com/v1/models",
                headers=self.headers
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False

    def get_available_models(self):
        """Return list of available models with custom names."""
        return [
            {"id": model_id, "name": f"Aleph {model_id}"} 
            for model_id in self.model_mapping.keys()
        ]

    def pipe(self, user_message: str, model_id: str, messages: List[Dict[str, Any]], body: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message through the Anthropic API."""
        try:
            # Map model ID
            anthropic_model_id = self.model_mapping.get(model_id)
            if not anthropic_model_id:
                raise ValueError(f"Unknown model: {model_id}")
            
            # Process messages
            processed_messages = [
                {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
                for message in messages
            ]

            # Prepare payload
            payload = {
                "model": anthropic_model_id,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.7),
                "stream": body.get("stream", False)
            }

            logger.info(f"Sending request to Anthropic API with payload: {json.dumps(payload, indent=2)}")
            
            # Send request
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()

            return response.json()
        
        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}")
            return {"error": str(e)}
