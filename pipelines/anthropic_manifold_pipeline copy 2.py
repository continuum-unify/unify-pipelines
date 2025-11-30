"""
title: Anthropic Manifold Pipeline
author: justinh-rahb, sriparashiva (corrected version)
date: 2024-06-20
version: 1.5
license: MIT
description: A pipeline for generating text and processing images using the Anthropic API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
import json
import base64
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import sseclient

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"

        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")}
        )
        self.url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()

    def update_headers(self):
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY
        }

    def get_anthropic_models(self):
        return [
            # Claude 3 models
            {"id": "claude-3-haiku-20240307", "name": "claude-3-haiku"},
            {"id": "claude-3-sonnet-20240229", "name": "claude-3-sonnet"},
            {"id": "claude-3-opus-20240229", "name": "claude-3-opus"},
            # Claude 3.5 models
            {"id": "claude-3-5-haiku-20241022", "name": "claude-3.5-haiku"},
            {"id": "claude-3-5-sonnet-20241022", "name": "claude-3.5-sonnet"},
            # Claude 3.7 models
            {"id": "claude-3-7-sonnet-20250219", "name": "claude-3.7-sonnet"},
            # Claude 4 models (latest)
            {"id": "claude-sonnet-4-20250514", "name": "claude-sonnet-4"},
            {"id": "claude-opus-4-20250514", "name": "claude-opus-4"},
        ]

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.update_headers()

    def pipelines(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_image(self, image_data):
        """
        Process image data for Anthropic API.
        Anthropic only supports base64-encoded images, not URLs.
        """
        if image_data["url"].startswith("data:image"):
            # Already base64 encoded
            mime_type, base64_data = image_data["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # URL image - need to download and convert to base64
            try:
                response = requests.get(image_data["url"], timeout=30)
                response.raise_for_status()
                
                # Get media type from response headers or infer from URL
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    media_type = 'image/jpeg'
                elif 'png' in content_type:
                    media_type = 'image/png'
                elif 'gif' in content_type:
                    media_type = 'image/gif'
                elif 'webp' in content_type:
                    media_type = 'image/webp'
                else:
                    # Try to infer from URL
                    url_lower = image_data["url"].lower()
                    if '.png' in url_lower:
                        media_type = 'image/png'
                    elif '.gif' in url_lower:
                        media_type = 'image/gif'
                    elif '.webp' in url_lower:
                        media_type = 'image/webp'
                    else:
                        media_type = 'image/jpeg'  # Default to JPEG
                
                base64_data = base64.b64encode(response.content).decode('utf-8')
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download image from URL: {e}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Remove unnecessary keys
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            system_message, messages = pop_system_message(messages)

            processed_messages = []
            image_count = 0
            total_image_size = 0

            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            if image_count >= 20:
                                raise ValueError("Maximum of 20 images per API call exceeded")

                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)

                            # Calculate image size for base64 data
                            if processed_image["source"]["type"] == "base64":
                                # Base64 string length * 3/4 gives approximate byte size
                                image_size = len(processed_image["source"]["data"]) * 3 / 4
                            else:
                                image_size = 0

                            total_image_size += image_size
                            if total_image_size > 100 * 1024 * 1024:  # 100 MB limit
                                raise ValueError("Total size of images exceeds 100 MB limit")

                            image_count += 1
                else:
                    processed_content = [{"type": "text", "text": message.get("content", "")}]

                processed_messages.append({"role": message["role"], "content": processed_content})

            # Prepare the payload
            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.8),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop_sequences": body.get("stop", []),
                **({"system": str(system_message)} if system_message else {}),
                "stream": body.get("stream", False),
            }

            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return self.get_completion(payload)
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, payload: dict) -> Generator:
        try:
            response = requests.post(
                self.url, 
                headers=self.headers, 
                json=payload, 
                stream=True,
                timeout=120
            )

            if response.status_code == 200:
                client = sseclient.SSEClient(response)
                for event in client.events():
                    try:
                        data = json.loads(event.data)
                        if data["type"] == "content_block_start":
                            if "content_block" in data and "text" in data["content_block"]:
                                yield data["content_block"]["text"]
                        elif data["type"] == "content_block_delta":
                            if "delta" in data and "text" in data["delta"]:
                                yield data["delta"]["text"]
                        elif data["type"] == "message_stop":
                            break
                        elif data["type"] == "error":
                            raise Exception(f"Anthropic API error: {data.get('error', {}).get('message', 'Unknown error')}")
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {event.data}")
                    except KeyError as e:
                        print(f"Unexpected data structure: {e}")
                        print(f"Full data: {data}")
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", response.text)
                except:
                    pass
                raise Exception(f"Error {response.status_code}: {error_detail}")
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def get_completion(self, payload: dict) -> str:
        try:
            response = requests.post(
                self.url, 
                headers=self.headers, 
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                res = response.json()
                return res["content"][0]["text"] if "content" in res and res["content"] else ""
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", response.text)
                except:
                    pass
                raise Exception(f"Error {response.status_code}: {error_detail}")
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")