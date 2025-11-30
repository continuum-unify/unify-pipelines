"""
title: ResetData Llama Manifold Pipeline
author: Continuum
date: 2024-12-01
version: 1.3
license: MIT
description: A pipeline for ResetData hosted Llama models with configurable token limits.
requirements: requests
environment_variables: RESETDATA_API_KEY
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field


class Pipeline:
    class Valves(BaseModel):
        RESETDATA_API_KEY: str = Field(default="", description="Your ResetData API key")
        RESETDATA_BASE_URL: str = Field(default="https://models.au-syd.resetdata.ai/v1", description="ResetData API base URL")
        DEFAULT_MAX_TOKENS: int = Field(default=32768, description="Default maximum tokens for responses (up to 128000 for Maverick)")
        DEFAULT_TEMPERATURE: float = Field(default=0.7, description="Default temperature (0.0-1.0)")

    def __init__(self):
        self.type = "manifold"
        self.id = "resetdata"
        self.name = "resetdata/"

        self.valves = self.Valves(
            RESETDATA_API_KEY=os.getenv("RESETDATA_API_KEY", ""),
            RESETDATA_BASE_URL=os.getenv("RESETDATA_BASE_URL", "https://models.au-syd.resetdata.ai/v1"),
            DEFAULT_MAX_TOKENS=int(os.getenv("RESETDATA_MAX_TOKENS", "32768")),
            DEFAULT_TEMPERATURE=float(os.getenv("RESETDATA_TEMPERATURE", "0.7"))
        )
        
        # Map simplified IDs to actual ResetData model IDs
        self.model_map = {
            "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct:shared",
            "llama-3.2-vision": "meta/llama-3.2-11b-vision-instruct:shared",
            "llama-3.1-8b": "meta/llama-3.1-8b-instruct:shared",
        }

    def get_resetdata_models(self):
        """
        Define available ResetData models with simplified IDs.
        Using simple IDs to avoid URL encoding issues with special characters.
        """
        return [
            {
                "id": "llama-4-maverick",
                "name": "Llama 4 Maverick 17B (1M context)"
            },
            {
                "id": "llama-3.2-vision",
                "name": "Llama 3.2 11B Vision Instruct"
            },
            {
                "id": "llama-3.1-8b",
                "name": "Llama 3.1 8B Instruct (Fast)"
            },
        ]

    def get_actual_model_id(self, simplified_id: str) -> str:
        """
        Convert simplified model ID to actual ResetData model ID.
        """
        return self.model_map.get(simplified_id, simplified_id)

    def get_model_config(self, model_id: str) -> dict:
        """
        Return model-specific configuration including max tokens.
        """
        configs = {
            "llama-4-maverick": {
                "max_tokens": self.valves.DEFAULT_MAX_TOKENS,
                "context_window": 1000000,
                "supports_vision": True,
            },
            "llama-3.2-vision": {
                "max_tokens": min(self.valves.DEFAULT_MAX_TOKENS, 8192),
                "context_window": 128000,
                "supports_vision": True,
            },
            "llama-3.1-8b": {
                "max_tokens": min(self.valves.DEFAULT_MAX_TOKENS, 8192),
                "context_window": 128000,
                "supports_vision": False,
            },
        }
        return configs.get(model_id, {
            "max_tokens": 8192,
            "context_window": 128000,
            "supports_vision": False,
        })

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        print(f"Default max_tokens: {self.valves.DEFAULT_MAX_TOKENS}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
        print(f"Updated max_tokens: {self.valves.DEFAULT_MAX_TOKENS}")
        pass

    def pipelines(self) -> List[dict]:
        return self.get_resetdata_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Remove unnecessary keys that Open WebUI adds
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            # Get model-specific configuration
            model_config = self.get_model_config(model_id)
            
            # Convert simplified ID to actual ResetData model ID
            actual_model_id = self.get_actual_model_id(model_id)
            
            api_key = self.valves.RESETDATA_API_KEY
            
            if not api_key:
                return "Error: No API key configured. Please set RESETDATA_API_KEY in the pipeline valves or environment."

            # Extract system message if present
            system_message = None
            filtered_messages = []
            for message in messages:
                if message.get("role") == "system":
                    system_message = message.get("content", "")
                else:
                    filtered_messages.append(message)

            # Build the messages array for the API
            api_messages = []
            if system_message:
                api_messages.append({"role": "system", "content": system_message})
            api_messages.extend(filtered_messages)

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            # Prepare the payload with ACTUAL model ID for ResetData API
            # Use body's max_tokens if provided, otherwise use valve default
            requested_max_tokens = body.get("max_tokens", model_config["max_tokens"])
            
            payload = {
                "model": actual_model_id,
                "messages": api_messages,
                "max_tokens": requested_max_tokens,
                "temperature": body.get("temperature", self.valves.DEFAULT_TEMPERATURE),
                "top_p": body.get("top_p", 0.9),
                "stream": body.get("stream", True),
            }

            # Add optional parameters if provided
            if "stop" in body and body["stop"]:
                payload["stop"] = body["stop"]
            
            if "frequency_penalty" in body:
                payload["frequency_penalty"] = body["frequency_penalty"]
            
            if "presence_penalty" in body:
                payload["presence_penalty"] = body["presence_penalty"]

            url = f"{self.valves.RESETDATA_BASE_URL}/chat/completions"

            if body.get("stream", True):
                return self.stream_response(url, headers, payload)
            else:
                return self.get_completion(url, headers, payload)

        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, url: str, headers: dict, payload: dict) -> Generator:
        """Handle streaming responses from the API."""
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=600
            )

            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_detail = error_json["error"].get("message", response.text)
                except:
                    pass
                raise Exception(f"Error {response.status_code}: {error_detail}")

        except requests.exceptions.Timeout:
            raise Exception("Request timed out. The model may be processing a large request.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def get_completion(self, url: str, headers: dict, payload: dict) -> str:
        """Handle non-streaming responses from the API."""
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=600
            )

            if response.status_code == 200:
                res = response.json()
                if "choices" in res and len(res["choices"]) > 0:
                    return res["choices"][0].get("message", {}).get("content", "")
                return ""
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_detail = error_json["error"].get("message", response.text)
                except:
                    pass
                raise Exception(f"Error {response.status_code}: {error_detail}")

        except requests.exceptions.Timeout:
            raise Exception("Request timed out. The model may be processing a large request.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")