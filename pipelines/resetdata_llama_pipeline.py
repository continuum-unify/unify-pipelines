"""
title: ResetData Llama Manifold Pipeline
author: Continuum
date: 2024-12-01
version: 1.0
license: MIT
description: A pipeline for ResetData hosted Llama models with proper token limits.
requirements: requests
environment_variables: RESETDATA_API_KEY
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        RESETDATA_API_KEY: str = ""
        RESETDATA_BASE_URL: str = "https://models.au-syd.resetdata.ai/v1"

    def __init__(self):
        self.type = "manifold"
        self.id = "resetdata"
        self.name = "resetdata/"

        self.valves = self.Valves(
            RESETDATA_API_KEY=os.getenv("RESETDATA_API_KEY", ""),
            RESETDATA_BASE_URL=os.getenv("RESETDATA_BASE_URL", "https://models.au-syd.resetdata.ai/v1")
        )

    def get_resetdata_models(self):
        """
        Define available ResetData models.
        Based on API response from https://models.au-syd.resetdata.ai/v1/models
        """
        return [
            # ============================================
            # Llama 4 - Latest and Most Capable
            # ============================================
            {
                "id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct:shared",
                "name": "Llama 4 Maverick 17B (1M context)"
            },
            
            # ============================================
            # Llama 3.2 - Vision Capable
            # ============================================
            {
                "id": "meta/llama-3.2-11b-vision-instruct:shared",
                "name": "Llama 3.2 11B Vision Instruct"
            },
            
            # ============================================
            # Llama 3.1 - Fast and Efficient
            # ============================================
            {
                "id": "meta/llama-3.1-8b-instruct:shared",
                "name": "Llama 3.1 8B Instruct (Fast)"
            },
            
            # ============================================
            # Embedding & Reranking Models (for RAG)
            # Note: These may not work for chat completions
            # ============================================
            # {
            #     "id": "nvidia/llama-3.2-nv-embedqa-1b-v2:shared",
            #     "name": "Llama 3.2 NV EmbedQA 1B V2"
            # },
            # {
            #     "id": "nvidia/llama-3.2-nv-rerankqa-1b-v2:shared",
            #     "name": "Llama 3.2 NV RerankQA 1B V2"
            # },
        ]

    def get_model_config(self, model_id: str) -> dict:
        """
        Return model-specific configuration including max tokens.
        """
        configs = {
            # Llama 4 Maverick - 1M context, large output
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct:shared": {
                "max_tokens": 32768,  # 32K default output
                "context_window": 1000000,  # 1M tokens
                "supports_vision": True,
            },
            # Llama 3.2 Vision
            "meta/llama-3.2-11b-vision-instruct:shared": {
                "max_tokens": 8192,
                "context_window": 128000,
                "supports_vision": True,
            },
            # Llama 3.1 8B
            "meta/llama-3.1-8b-instruct:shared": {
                "max_tokens": 8192,
                "context_window": 128000,
                "supports_vision": False,
            },
        }
        # Default config for unknown models
        return configs.get(model_id, {
            "max_tokens": 8192,
            "context_window": 128000,
            "supports_vision": False,
        })

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        print(f"on_valves_updated:{__name__}")
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
                "Authorization": f"Bearer {self.valves.RESETDATA_API_KEY}"
            }

            # Prepare the payload
            # Use body's max_tokens if provided, otherwise use model default
            requested_max_tokens = body.get("max_tokens", model_config["max_tokens"])
            
            payload = {
                "model": model_id,
                "messages": api_messages,
                "max_tokens": requested_max_tokens,
                "temperature": body.get("temperature", 0.7),
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
                timeout=600  # 10 minute timeout for long responses
            )

            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        # Skip empty lines and handle SSE format
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
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
                                # Skip malformed JSON
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
                timeout=600  # 10 minute timeout for long responses
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