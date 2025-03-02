"""
title: Aleph Manifold Pipeline
author: Your Name
date: 2024-03-02
version: 1.5
license: MIT
description: Streamlined pipeline for text and image processing using Anthropic's Claude API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import json
import base64
import requests
import logging
from typing import List, Dict, Any, Generator
from pydantic import BaseModel
import sseclient
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        MAX_IMAGE_SIZE: int = 20 * 1024 * 1024  # 20MB limit

    def __init__(self):
        """Initialize the Aleph pipeline."""
        self.type = "filter"
        self.name = "aleph_manifold"
        
        # Define model mapping
        self.model_mapping = {
            "1": "claude-3-7-sonnet-20250219",
            "2": "claude-3-5-haiku-20241022",
            "3": "claude-3-5-sonnet-20241022",
            "4": "claude-3-opus-20240229",
            "5": "claude-3-sonnet-20240229"
        }
        
        # Initialize valves
        self.valves = self.Valves(
            ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", "")
        )
        
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": self.valves.ANTHROPIC_API_KEY
        }
        
        logger.info("Aleph pipeline initialized")

    async def on_startup(self):
        """Startup tasks."""
        if not self.valves.ANTHROPIC_API_KEY:
            logger.warning("No Anthropic API key provided")
        logger.info("Aleph pipeline started")

    async def on_shutdown(self):
        """Cleanup tasks."""
        logger.info("Aleph pipeline shut down")

    async def inlet(self, body: Dict[str, Any], user: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process the request before sending to the LLM."""
        try:
            # Process any images in messages
            if "messages" in body:
                for i, msg in enumerate(body["messages"]):
                    if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        # Check for attachments
                        attachments = body.get("attachments", [])
                        if attachments:
                            body["messages"][i] = self._process_message_with_images(msg, attachments)
            
            # Map model ID if needed
            if "model" in body and body["model"] in self.model_mapping:
                body["model"] = self.model_mapping[body["model"]]
            
            return body
        except Exception as e:
            logger.error(f"Inlet error: {str(e)}")
            return body

    def _process_message_with_images(self, message: Dict[str, Any], attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add images to the message content."""
        if not attachments:
            return message
            
        new_content = []
        
        # Add the text content
        content = message.get("content", "")
        if content:
            new_content.append({
                "type": "text",
                "text": content
            })
        
        # Add images
        for attachment in attachments:
            if attachment.get("type", "").startswith("image/"):
                try:
                    image_data = attachment.get("data", "")
                    if image_data and image_data.startswith("data:"):
                        # Handle data URI
                        _, encoded = image_data.split(",", 1)
                        img_str = encoded
                    else:
                        # No valid image data
                        continue
                        
                    new_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": attachment["type"],
                            "data": img_str
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
        
        # If we have content, return the structured message
        if new_content:
            return {
                "role": message["role"],
                "content": new_content
            }
        
        # Otherwise return the original message
        return message

    async def outlet(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the LLM response."""
        # Add pipeline identifier to the response
        if "id" not in response:
            response["id"] = f"aleph_{response.get('id', 'response')}"
        return response

    def pipe(self, user_message: str, model_id: str, messages: List[Dict[str, Any]], body: Dict[str, Any]) -> Any:
        """Main pipeline processing logic."""
        try:
            # Map the model ID
            anthropic_model_id = self.model_mapping.get(model_id)
            if not anthropic_model_id:
                return {"error": f"Unknown model: {model_id}"}

            # Format messages for Anthropic API
            formatted_messages = []
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                elif isinstance(msg.get("content"), list):
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Prepare the API request
            payload = {
                "model": anthropic_model_id,
                "messages": formatted_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.7),
                "stream": body.get("stream", False)
            }

            # Handle streaming vs standard requests
            if body.get("stream", False):
                return self._stream_response(payload)
            else:
                return self._get_completion(payload)

        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}")
            return {"error": str(e)}

    def _stream_response(self, payload: Dict[str, Any]) -> Generator:
        """Handle streaming responses."""
        try:
            response = requests.post(
                self.url, 
                headers=self.headers, 
                json=payload, 
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                yield f"Error: {response.status_code}"
                return
                
            client = sseclient.SSEClient(response)
            for event in client.events():
                if not event.data:
                    continue
                    
                try:
                    data = json.loads(event.data)
                    if data.get("type") == "content_block_delta":
                        yield data["delta"]["text"]
                    elif data.get("type") == "message_stop":
                        break
                except Exception:
                    continue

        except Exception as e:
            yield f"Streaming error: {str(e)}"

    def _get_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard responses."""
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "details": response.text}
                
            return response.json()
                
        except Exception as e:
            return {"error": f"API request error: {str(e)}"}