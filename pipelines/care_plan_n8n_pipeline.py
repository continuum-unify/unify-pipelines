"""
title: Care Plan Assistant via n8n
author: Continuum Labs
version: 1.2.0
license: MIT
description: AI-powered care plan assistant that integrates with n8n workflow automation for aged care facilities. Guides users through 6 sections to create comprehensive, person-centred care plans.
requirements: requests
environment_variables: N8N_WEBHOOK_URL, N8N_AUTH_TOKEN
"""

from typing import List, Union, Generator, Iterator, Optional
import requests
import os
import json
import uuid
import hashlib
import time
from pydantic import BaseModel, Field


class Pipeline:
    """
    Care Plan Assistant Pipeline v1.2.0
    
    This pipeline connects Open WebUI to an n8n workflow that processes
    care plan requests using AI agents with memory for multi-turn conversations.
    
    Features:
    - Unique session IDs per conversation
    - Guided 6-section care plan creation
    - Formatted care plan output
    - Debug logging for troubleshooting
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
            default=180,
            description="Timeout in seconds for n8n requests (longer for care plan generation)"
        )
        
        enable_debug_logging: bool = Field(
            default=True,
            description="Enable detailed debug logging"
        )
        
        fallback_message: str = Field(
            default="I apologize, but I'm having trouble connecting to the care plan service. Please try again or contact your administrator.",
            description="Message to show when n8n is unavailable"
        )

    def __init__(self):
        """Initialize the pipeline"""
        self.type = "manifold"
        self.id = "care_plan_n8n"
        self.name = "Care Plan Assistant (n8n)"
        self.valves = self.Valves()
        # Store session IDs to maintain consistency within a conversation
        self._session_cache = {}

    async def on_startup(self):
        """Called when the pipeline is loaded"""
        print(f"✅ {self.name} pipeline initialized (v1.2.0)")
        print(f"📡 n8n webhook: {self.valves.n8n_webhook_url}")
        print(f"⏱️ Timeout: {self.valves.request_timeout}s")

    async def on_shutdown(self):
        """Called when the pipeline is unloaded"""
        print(f"👋 {self.name} pipeline shutting down")
        self._session_cache.clear()

    def pipelines(self) -> List[dict]:
        """Define available models/pipelines"""
        return [
            {
                "id": "care-plan-assistant",
                "name": "Care Plan Assistant",
                "description": "AI assistant for creating comprehensive, person-centred aged care plans"
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
        """
        
        if self.valves.enable_debug_logging:
            print(f"\n{'='*60}")
            print(f"📨 Care Plan Assistant - New Message")
            print(f"{'='*60}")
            print(f"Message: {user_message[:100]}...")
            print(f"Message count in history: {len(messages)}")
        
        try:
            # Prepare payload for n8n
            payload = self._prepare_payload(
                user_message=user_message,
                messages=messages,
                model_id=model_id,
                body=body
            )
            
            if self.valves.enable_debug_logging:
                print(f"🆔 Session ID: {payload['chat']['id']}")
                print(f"🔄 Sending to: {self.valves.n8n_webhook_url}")
            
            # Send request to n8n
            response = requests.post(
                self.valves.n8n_webhook_url,
                headers={
                    "Authorization": self.valves.n8n_auth_token,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.valves.request_timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            if self.valves.enable_debug_logging:
                print(f"✅ Response received from n8n")
            
            # Extract AI response
            ai_response = self._extract_response(result)
            
            if not ai_response:
                print(f"⚠️ Empty response from n8n: {result}")
                return self.valves.fallback_message
            
            return ai_response
            
        except requests.exceptions.Timeout:
            print(f"⏱️ Request timed out after {self.valves.request_timeout}s")
            return "The care plan is taking longer than expected to generate. Please try again - if creating a full care plan, this may take up to a minute."
            
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Connection error: {str(e)}")
            return self.valves.fallback_message
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP error: {e.response.status_code}")
            if e.response.status_code == 401:
                return "Authentication error with the care plan service. Please contact your administrator."
            elif e.response.status_code == 404:
                return "The care plan service is not available. Please verify the n8n workflow is active."
            else:
                return f"Error communicating with care plan service (HTTP {e.response.status_code}). Please try again."
            
        except Exception as e:
            print(f"💥 Unexpected error: {type(e).__name__} - {str(e)}")
            return "An unexpected error occurred. Please try again or contact support."

    def _get_session_id(self, body: dict, messages: List[dict]) -> str:
        """
        Get or create a unique session ID for this conversation.
        
        The session ID must be:
        1. Unique per conversation (different chats get different IDs)
        2. Consistent within a conversation (same chat always gets same ID)
        
        Args:
            body: The request body from Open WebUI
            messages: The conversation messages
            
        Returns:
            A unique session identifier string
        """
        chat_id = None
        source = None
        
        # Try to get chat ID from various possible locations
        possible_sources = [
            ("body.chat_id", body.get("chat_id")),
            ("body.id", body.get("id")),
            ("body.session_id", body.get("session_id")),
            ("body.metadata.chat_id", body.get("metadata", {}).get("chat_id")),
            ("body.metadata.session_id", body.get("metadata", {}).get("session_id")),
        ]
        
        for src_name, src_value in possible_sources:
            if src_value and src_value != "default":
                chat_id = str(src_value)
                source = src_name
                break
        
        # If no chat ID found, generate a deterministic one
        if not chat_id:
            user_id = body.get("user", {}).get("id", "anonymous")
            
            # Create a deterministic session ID based on:
            # - User ID
            # - First user message content (to differentiate conversations)
            # - Approximate start time (rounded to 10-minute windows)
            
            first_user_message = ""
            for msg in messages:
                if msg.get("role") == "user":
                    first_user_message = msg.get("content", "")[:200]
                    break
            
            if first_user_message:
                # Hash the first message to create consistent session ID
                hash_input = f"{user_id}:{first_user_message}"
                chat_id = "cp_" + hashlib.sha256(hash_input.encode()).hexdigest()[:12]
                source = "generated_from_first_message"
            else:
                # Fallback: use user ID + timestamp window
                time_window = int(time.time() // 600)  # 10-minute windows
                hash_input = f"{user_id}:{time_window}"
                chat_id = "cp_" + hashlib.sha256(hash_input.encode()).hexdigest()[:12]
                source = "generated_from_time_window"
        
        if self.valves.enable_debug_logging:
            print(f"🔑 Session ID: {chat_id} (source: {source})")
        
        return chat_id

    def _prepare_payload(
        self,
        user_message: str,
        messages: List[dict],
        model_id: str,
        body: dict
    ) -> dict:
        """Prepare the payload to send to n8n"""
        
        session_id = self._get_session_id(body, messages)
        
        # Get user info
        user_info = body.get("user", {})
        
        return {
            "message": user_message,
            "messages": messages,
            "user": {
                "id": user_info.get("id", "anonymous"),
                "name": user_info.get("name", "Care Staff"),
                "email": user_info.get("email", "")
            },
            "chat": {
                "id": session_id,
                "model": model_id
            },
            "metadata": {
                "timestamp": body.get("timestamp"),
                "stream": body.get("stream", False),
                "pipeline": self.name,
                "version": "1.2.0",
                "message_count": len(messages)
            }
        }

    def _extract_response(self, result: dict) -> str:
        """Extract the AI response from n8n's response"""
        
        if isinstance(result, str):
            return result
        
        if isinstance(result, dict):
            # Try common response keys in order of likelihood
            for key in ["output", "response", "text", "content", "message"]:
                if key in result and result[key]:
                    value = result[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        # Recursively extract from nested dict
                        return self._extract_response(value)
            
            # Check for nested data structure
            if "data" in result:
                return self._extract_response(result["data"])
        
        if isinstance(result, list) and len(result) > 0:
            return self._extract_response(result[0])
        
        return ""