"""
title: Care Plan Assistant via n8n
author: Continuum Labs
version: 1.0.0
license: MIT
description: AI-powered care plan assistant that integrates with n8n workflow automation for aged care facilities
requirements: requests
environment_variables: N8N_WEBHOOK_URL, N8N_AUTH_TOKEN
"""

from typing import List, Union, Generator, Iterator, Optional
import requests
import os
import json
from pydantic import BaseModel, Field


class Pipeline:
    """
    Care Plan Assistant Pipeline
    
    This pipeline connects Open WebUI to an n8n workflow that processes
    care plan requests using AI agents with access to Aged Care Act compliance,
    Quality Standards, and best practices documentation.
    """
    
    class Valves(BaseModel):
        """
        Configurable settings for the pipeline.
        These can be adjusted from the Open WebUI admin interface.
        """
        n8n_webhook_url: str = Field(
            default=os.getenv(
                "N8N_WEBHOOK_URL",
                "https://n8n.continuumlabs.tech/webhook/care-plan-agent"
            ),
            description="n8n webhook endpoint URL"
        )
        
        n8n_auth_token: str = Field(
            default=os.getenv(
                "N8N_AUTH_TOKEN",
                "Bearer n8n_openwebui_secret_2025"
            ),
            description="Authentication token for n8n webhook (include 'Bearer ' prefix)"
        )
        
        request_timeout: int = Field(
            default=60,
            description="Timeout in seconds for n8n requests"
        )
        
        enable_debug_logging: bool = Field(
            default=False,
            description="Enable detailed debug logging"
        )
        
        fallback_message: str = Field(
            default="I apologize, but I'm having trouble connecting to the care plan service. Please try again or contact your administrator.",
            description="Message to show when n8n is unavailable"
        )

    def __init__(self):
        """Initialize the pipeline"""
        self.type = "manifold"  # Acts as a model provider
        self.id = "care_plan_n8n"
        self.name = "Care Plan Assistant (n8n)"
        self.valves = self.Valves()

    async def on_startup(self):
        """Called when the pipeline is loaded"""
        print(f"✅ {self.name} pipeline initialized")
        print(f"📡 n8n webhook: {self.valves.n8n_webhook_url}")
        
        # Test connection to n8n
        try:
            response = requests.get(
                self.valves.n8n_webhook_url.replace('/webhook/', '/webhook-test/'),
                timeout=5
            )
            if response.status_code == 200:
                print("✅ n8n connection test successful")
            else:
                print(f"⚠️ n8n connection test returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not connect to n8n: {str(e)}")

    async def on_shutdown(self):
        """Called when the pipeline is unloaded"""
        print(f"👋 {self.name} pipeline shutting down")

    def pipelines(self) -> List[dict]:
        """
        Define available models/pipelines
        """
        return [
            {
                "id": "care-plan-assistant",
                "name": "Care Plan Assistant",
                "description": "AI assistant for creating and managing aged care plans with compliance checking"
            }
        ]

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function - processes each message
        
        Args:
            user_message: The latest user message
            model_id: The model ID being used
            messages: Full conversation history
            body: Complete request body from Open WebUI
        
        Returns:
            AI response as string
        """
        
        if self.valves.enable_debug_logging:
            print(f"📨 Processing message: {user_message[:100]}...")
            print(f"📋 Message count: {len(messages)}")
        
        try:
            # Prepare payload for n8n
            payload = self._prepare_payload(
                user_message=user_message,
                messages=messages,
                model_id=model_id,
                body=body
            )
            
            # Send request to n8n webhook
            if self.valves.enable_debug_logging:
                print(f"🔄 Sending to n8n: {self.valves.n8n_webhook_url}")
            
            response = requests.post(
                self.valves.n8n_webhook_url,
                headers={
                    "Authorization": self.valves.n8n_auth_token,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.valves.request_timeout
            )
            
            # Handle response
            response.raise_for_status()
            result = response.json()
            
            if self.valves.enable_debug_logging:
                print(f"✅ Received response from n8n")
            
            # Extract AI response
            ai_response = self._extract_response(result)
            
            if not ai_response:
                print(f"⚠️ Empty response from n8n: {result}")
                return self.valves.fallback_message
            
            return ai_response
            
        except requests.exceptions.Timeout:
            error_msg = f"⏱️ Request to n8n timed out after {self.valves.request_timeout} seconds"
            print(error_msg)
            return "The request is taking longer than expected. Please try again with a simpler question or contact support."
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"🔌 Connection error to n8n: {str(e)}"
            print(error_msg)
            return self.valves.fallback_message
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"❌ HTTP error from n8n: {e.response.status_code}"
            print(error_msg)
            
            if e.response.status_code == 401:
                return "Authentication error with the care plan service. Please contact your administrator."
            elif e.response.status_code == 404:
                return "The care plan service endpoint was not found. Please verify the n8n workflow is active."
            else:
                return f"Error communicating with care plan service (HTTP {e.response.status_code}). Please try again."
            
        except Exception as e:
            error_msg = f"💥 Unexpected error: {type(e).__name__} - {str(e)}"
            print(error_msg)
            return f"An unexpected error occurred. Please try again or contact support."

    def _prepare_payload(
        self,
        user_message: str,
        messages: List[dict],
        model_id: str,
        body: dict
    ) -> dict:
        """
        Prepare the payload to send to n8n
        
        Args:
            user_message: Latest user message
            messages: Full conversation history
            model_id: Model identifier
            body: Original request body
        
        Returns:
            Formatted payload dictionary
        """
        return {
            "message": user_message,
            "messages": messages,
            "user": {
                "id": body.get("user", {}).get("id", "anonymous"),
                "name": body.get("user", {}).get("name", "User"),
                "email": body.get("user", {}).get("email", "")
            },
            "chat": {
                "id": body.get("chat_id", "default"),
                "model": model_id
            },
            "metadata": {
                "timestamp": body.get("timestamp"),
                "stream": body.get("stream", False),
                "pipeline": self.name,
                "version": "1.0.0"
            }
        }

    def _extract_response(self, result: dict) -> str:
        """
        Extract the AI response from n8n's response
        
        Args:
            result: JSON response from n8n
        
        Returns:
            Extracted response text
        """
        # Try different possible response structures
        if isinstance(result, str):
            return result
        
        if isinstance(result, dict):
            # Try common response keys
            for key in ["response", "output", "text", "content", "message"]:
                if key in result and result[key]:
                    return str(result[key])
            
            # If it's a nested structure, try to find text
            if "data" in result and isinstance(result["data"], dict):
                return self._extract_response(result["data"])
        
        # If we can't find a response, return empty string
        return ""