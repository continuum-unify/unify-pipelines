"""
title: Anthropic Manifold Pipeline (Enhanced)
author: Continuum Unify
version: 2.0.2
requirements: sseclient-py, requests
description: Enhanced Claude models integration with extended thinking, prompt caching, and token tracking
requirements: sseclient-py, requests

Features:
- Extended thinking support for complex reasoning (Claude 3.7+)
- Updated Claude 4.5 models with 64K output token support
- Prompt caching for 90% cost reduction on repeated system prompts
- Token usage tracking for cost monitoring
- Data sovereignty warnings for non-Australian processing

Changelog:
- v2.0.1: Fixed beta headers for extended thinking and prompt caching, updated API version
- v2.0.0: Added extended thinking, increased token limits, Claude 4.5 updates
- v1.0.0: Initial pipeline with basic Claude integration
"""

import base64
import json
import requests
import sseclient
from typing import Generator, List, Optional, Union
from pydantic import BaseModel, Field


class Pipeline:
    """
    Anthropic Manifold Pipeline - Translates OpenAI-format requests to Anthropic API format
    and exposes all Claude models as selectable options in Open WebUI.
    """

    class Valves(BaseModel):
        """Configuration exposed in Admin Panel → Settings → Pipelines"""
        
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Anthropic API key (sk-ant-...)"
        )
        ANTHROPIC_API_URL: str = Field(
            default="https://api.anthropic.com/v1/messages",
            description="Anthropic Messages API endpoint"
        )
        
        # Extended Thinking Configuration
        ENABLE_EXTENDED_THINKING: bool = Field(
            default=True,
            description="Enable extended thinking for supported models (Claude 3.7+)"
        )
        DEFAULT_THINKING_BUDGET: int = Field(
            default=16000,
            description="Default token budget for extended thinking (1024-128000)"
        )
        
        # Prompt Caching Configuration
        ENABLE_PROMPT_CACHING: bool = Field(
            default=True,
            description="Enable prompt caching for cost optimization (90% reduction)"
        )
        MIN_CACHE_TOKENS: int = Field(
            default=1024,
            description="Minimum system prompt length to enable caching"
        )
        
        # Token Tracking Configuration
        ENABLE_TOKEN_TRACKING: bool = Field(
            default=True,
            description="Track and log token usage for cost monitoring"
        )
        
        # Data Sovereignty Warning
        SHOW_SOVEREIGNTY_WARNING: bool = Field(
            default=True,
            description="Show warning that data is processed outside Australia"
        )

    def __init__(self):
        self.type = "manifold"  # Exposes multiple models
        self.id = "anthropic"
        self.name = "anthropic/"
        self.valves = self.Valves()
        
        # Token usage accumulator (for tracking)
        self.session_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "thinking_tokens": 0
        }

    def _get_headers(self, use_thinking: bool = False, use_caching: bool = False) -> dict:
        """
        Build headers for Anthropic API requests.
        
        Beta features require specific beta headers to be included.
        Multiple beta features can be combined with comma separation.
        """
        headers = {
            "anthropic-version": "2024-10-22",  # Updated for Claude 4.x features
            "content-type": "application/json",
            "x-api-key": self.valves.ANTHROPIC_API_KEY,
        }
        
        # Build beta header for optional features
        beta_features = []
        
        if use_thinking:
            beta_features.append("interleaved-thinking-2025-05-14")
        
        if use_caching:
            beta_features.append("prompt-caching-2024-07-31")
        
        if beta_features:
            headers["anthropic-beta"] = ",".join(beta_features)
        
        return headers

    def get_anthropic_models(self) -> List[dict]:
        """
        Returns all available Claude models with metadata.
        
        Model naming convention:
        - claude-{family}-{version}-{date}
        - Latest alias points to most recent stable version
        """
        return [
            # ═══════════════════════════════════════════════════════════════════
            # CLAUDE 4.5 MODELS (Latest - September-November 2025)
            # - 200K context (1M beta available)
            # - 64K max output tokens
            # - Extended thinking support
            # - Vision support
            # ═══════════════════════════════════════════════════════════════════
            {
                "id": "claude-sonnet-4-5-20250929",
                "name": "Claude Sonnet 4.5 (Recommended)",
                "description": "Best balance of intelligence and speed. $3/$15 per MTok.",
                "context_window": 200000,
                "max_output": 65536,
                "supports_thinking": True,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "January 2025"
            },
            {
                "id": "claude-haiku-4-5-20251001",
                "name": "Claude Haiku 4.5 (Fastest)",
                "description": "Fastest responses, cost-effective. $1/$5 per MTok.",
                "context_window": 200000,
                "max_output": 65536,
                "supports_thinking": True,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "February 2025"
            },
            {
                "id": "claude-opus-4-5-20251101",
                "name": "Claude Opus 4.5 (Premium Intelligence)",
                "description": "Maximum capability for complex tasks. $5/$25 per MTok.",
                "context_window": 200000,
                "max_output": 65536,
                "supports_thinking": True,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "May 2025"
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # CLAUDE 4 MODELS (May 2025)
            # ═══════════════════════════════════════════════════════════════════
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "description": "Claude 4 generation Sonnet.",
                "context_window": 200000,
                "max_output": 65536,
                "supports_thinking": True,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "April 2024"
            },
            {
                "id": "claude-opus-4-20250514",
                "name": "Claude Opus 4",
                "description": "Claude 4 generation premium model.",
                "context_window": 200000,
                "max_output": 65536,
                "supports_thinking": True,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "April 2024"
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # CLAUDE 3.5 MODELS (Legacy - No extended thinking)
            # ═══════════════════════════════════════════════════════════════════
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet (Legacy)",
                "description": "Previous generation. Use 4.5 for better results.",
                "context_window": 200000,
                "max_output": 8192,
                "supports_thinking": False,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "April 2024"
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku (Legacy)",
                "description": "Previous generation fast model.",
                "context_window": 200000,
                "max_output": 8192,
                "supports_thinking": False,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "April 2024"
            },
            
            # ═══════════════════════════════════════════════════════════════════
            # CLAUDE 3 MODELS (Legacy - Limited capabilities)
            # ═══════════════════════════════════════════════════════════════════
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus (Legacy)",
                "description": "Previous generation premium. Consider Opus 4.5.",
                "context_window": 200000,
                "max_output": 4096,
                "supports_thinking": False,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "August 2023"
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet (Legacy)",
                "description": "Previous generation balanced model.",
                "context_window": 200000,
                "max_output": 4096,
                "supports_thinking": False,
                "supports_vision": True,
                "supports_caching": True,
                "knowledge_cutoff": "August 2023"
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "Claude 3 Haiku (Legacy)",
                "description": "Previous generation fast model.",
                "context_window": 200000,
                "max_output": 4096,
                "supports_thinking": False,
                "supports_vision": False,
                "supports_caching": True,
                "knowledge_cutoff": "August 2023"
            },
        ]

    def pipelines(self) -> List[dict]:
        """
        Called by Open WebUI to get available models.
        Returns models in format expected by the UI.
        """
        return [{"id": m["id"], "name": m["name"]} for m in self.get_anthropic_models()]

    def get_model_config(self, model_id: str) -> dict:
        """Get configuration for a specific model by ID."""
        for model in self.get_anthropic_models():
            if model["id"] == model_id:
                return model
        # Default config for unknown models
        return {
            "max_output": 8192,
            "supports_thinking": False,
            "supports_vision": True,
            "supports_caching": True
        }

    def get_default_max_tokens(self, model_id: str) -> int:
        """
        Returns appropriate default max_tokens based on model capabilities.
        
        Claude 4.5 models support up to 64K output tokens.
        Older models have lower limits.
        """
        config = self.get_model_config(model_id)
        # Return a sensible default (not the maximum) to avoid unnecessary costs
        max_output = config.get("max_output", 8192)
        
        if max_output >= 65536:
            return 16384  # Default 16K for 4.5 models (can be increased by user)
        elif max_output >= 16384:
            return 8192   # Default 8K for 3.7 models
        elif max_output >= 8192:
            return 8192   # Default 8K for 3.5 models
        else:
            return 4096   # Default 4K for legacy models

    def supports_extended_thinking(self, model_id: str) -> bool:
        """Check if model supports extended thinking feature."""
        config = self.get_model_config(model_id)
        return config.get("supports_thinking", False)

    def process_image(self, image_data: dict) -> dict:
        """
        Converts image data to Anthropic's required format.
        
        Anthropic requires base64-encoded images (not URLs like OpenAI).
        Handles both data URLs and external URLs.
        """
        image_url = image_data.get("url", "")
        
        if image_url.startswith("data:image"):
            # Already a data URL - extract components
            try:
                # Format: data:image/jpeg;base64,/9j/4AAQ...
                header, base64_data = image_url.split(",", 1)
                media_type = header.split(":")[1].split(";")[0]
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    },
                }
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid image data URL format: {e}")
        else:
            # External URL - download and convert to base64
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Determine media type from Content-Type header or URL
                content_type = response.headers.get("Content-Type", "image/jpeg")
                if ";" in content_type:
                    content_type = content_type.split(";")[0]
                
                base64_data = base64.b64encode(response.content).decode("utf-8")
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content_type,
                        "data": base64_data,
                    },
                }
            except requests.RequestException as e:
                raise ValueError(f"Failed to download image from {image_url}: {e}")

    def process_messages(self, messages: List[dict]) -> tuple:
        """
        Processes messages from OpenAI format to Anthropic format.
        
        Key transformations:
        1. Extract system message (Anthropic handles it separately)
        2. Convert image_url to base64 image format
        3. Maintain conversation structure
        
        Returns: (system_message, processed_messages)
        """
        system_message = None
        processed_messages = []
        image_count = 0
        max_images = 20  # Anthropic limit
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Extract system message
            if role == "system":
                if isinstance(content, list):
                    system_message = " ".join(
                        item.get("text", "") for item in content if item.get("type") == "text"
                    )
                else:
                    system_message = str(content)
                continue
            
            # Process user/assistant messages
            if isinstance(content, list):
                # Multi-modal content (text + images)
                processed_content = []
                for item in content:
                    if item.get("type") == "text":
                        processed_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image_url":
                        if image_count >= max_images:
                            raise ValueError(f"Maximum {max_images} images allowed per request")
                        processed_content.append(self.process_image(item.get("image_url", {})))
                        image_count += 1
                
                processed_messages.append({
                    "role": role if role in ["user", "assistant"] else "user",
                    "content": processed_content
                })
            else:
                # Text-only content
                processed_messages.append({
                    "role": role if role in ["user", "assistant"] else "user",
                    "content": str(content)
                })
        
        return system_message, processed_messages

    def build_system_message(self, system_message: Optional[str], use_caching: bool) -> Union[str, List[dict], None]:
        """
        Builds system message with optional caching.
        
        Prompt caching can reduce costs by 90% for repeated system prompts.
        Caching is applied when:
        - use_caching is True
        - System prompt exceeds MIN_CACHE_TOKENS (default 1024)
        """
        if not system_message:
            return None
        
        # Check if caching should be applied
        if use_caching and len(system_message) >= self.valves.MIN_CACHE_TOKENS:
            # Return as cacheable block
            return [{
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }]
        else:
            # Return as simple string
            return system_message

    def build_payload(
        self,
        model_id: str,
        messages: List[dict],
        system_message: Optional[str],
        body: dict,
        use_thinking: bool,
        use_caching: bool
    ) -> dict:
        """
        Constructs the API payload for Anthropic.
        
        Handles:
        - Model-specific max_tokens
        - Extended thinking configuration
        - Temperature settings (excludes top_p to avoid 400 error)
        - Stop sequences
        - Streaming configuration
        """
        model_config = self.get_model_config(model_id)
        
        # Determine max_tokens
        max_tokens = body.get("max_tokens")
        if not max_tokens:
            max_tokens = self.get_default_max_tokens(model_id)
        # Cap at model's maximum
        max_tokens = min(max_tokens, model_config.get("max_output", 8192))
        
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": body.get("stream", False),
        }
        
        # Add system message (with optional caching)
        system_content = self.build_system_message(system_message, use_caching)
        if system_content:
            payload["system"] = system_content
        
        # Temperature (IMPORTANT: Don't send both temperature AND top_p)
        # Anthropic returns 400 error if both are specified
        if "temperature" in body:
            payload["temperature"] = body["temperature"]
        else:
            payload["temperature"] = 0.7
        
        # Stop sequences
        if "stop" in body and body["stop"]:
            payload["stop_sequences"] = body["stop"]
        
        # Extended Thinking Configuration
        if use_thinking:
            thinking_budget = body.get("thinking_budget", self.valves.DEFAULT_THINKING_BUDGET)
            # Clamp to valid range
            thinking_budget = max(1024, min(thinking_budget, 128000))
            
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
            
            # When thinking is enabled, temperature must be 1
            # and max_tokens must be > thinking_budget
            payload["temperature"] = 1
            if payload["max_tokens"] <= thinking_budget:
                payload["max_tokens"] = thinking_budget + 8192
        
        return payload

    def stream_response(self, payload: dict, headers: dict) -> Generator:
        """
        Streams response from Anthropic API.
        
        Handles multiple event types:
        - message_start: Initial message metadata
        - content_block_start: Beginning of content block (text or thinking)
        - content_block_delta: Incremental content (text_delta, thinking_delta)
        - content_block_stop: End of content block
        - message_delta: Usage statistics
        - message_stop: End of message
        
        Extended thinking content is yielded with special formatting.
        """
        try:
            response = requests.post(
                self.valves.ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=300  # 5 minute timeout for long responses
            )
            response.raise_for_status()
            
            client = sseclient.SSEClient(response)
            current_block_type = None
            in_thinking = False
            
            for event in client.events():
                if event.data == "[DONE]":
                    break
                
                try:
                    data = json.loads(event.data)
                    event_type = data.get("type", "")
                    
                    if event_type == "message_start":
                        # Track input tokens from message_start
                        if self.valves.ENABLE_TOKEN_TRACKING:
                            message = data.get("message", {})
                            usage = message.get("usage", {})
                            self.session_usage["input_tokens"] += usage.get("input_tokens", 0)
                            self.session_usage["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)
                            self.session_usage["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
                    
                    elif event_type == "content_block_start":
                        block = data.get("content_block", {})
                        current_block_type = block.get("type", "text")
                        
                        if current_block_type == "thinking":
                            in_thinking = True
                            yield "\n<thinking>\n"
                    
                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type", "")
                        
                        if delta_type == "text_delta":
                            yield delta.get("text", "")
                        elif delta_type == "thinking_delta":
                            yield delta.get("thinking", "")
                    
                    elif event_type == "content_block_stop":
                        if in_thinking:
                            yield "\n</thinking>\n\n"
                            in_thinking = False
                        current_block_type = None
                    
                    elif event_type == "message_delta":
                        # Track usage statistics
                        if self.valves.ENABLE_TOKEN_TRACKING:
                            usage = data.get("usage", {})
                            self.session_usage["output_tokens"] += usage.get("output_tokens", 0)
                    
                    elif event_type == "error":
                        error = data.get("error", {})
                        raise Exception(f"Anthropic API error: {error.get('message', 'Unknown error')}")
                
                except json.JSONDecodeError:
                    continue
        
        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.text
            except:
                pass
            yield f"\n\n**HTTP Error {e.response.status_code}:** {str(e)}\n{error_body}"
        except requests.RequestException as e:
            yield f"\n\n**Error communicating with Anthropic API:** {str(e)}"

    def get_completion(self, payload: dict, headers: dict) -> str:
        """
        Gets non-streaming completion from Anthropic API.
        Returns the complete response text.
        """
        payload["stream"] = False
        
        try:
            response = requests.post(
                self.valves.ANTHROPIC_API_URL,
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Track usage
            if self.valves.ENABLE_TOKEN_TRACKING:
                usage = data.get("usage", {})
                self.session_usage["input_tokens"] += usage.get("input_tokens", 0)
                self.session_usage["output_tokens"] += usage.get("output_tokens", 0)
                self.session_usage["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)
                self.session_usage["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
            
            # Extract response text
            content = data.get("content", [])
            result_parts = []
            
            for block in content:
                if block.get("type") == "thinking":
                    result_parts.append(f"<thinking>\n{block.get('thinking', '')}\n</thinking>\n\n")
                elif block.get("type") == "text":
                    result_parts.append(block.get("text", ""))
            
            return "".join(result_parts)
        
        except requests.exceptions.HTTPError as e:
            error_body = ""
            try:
                error_body = e.response.text
            except:
                pass
            return f"**HTTP Error {e.response.status_code}:** {str(e)}\n{error_body}"
        except requests.RequestException as e:
            return f"**Error communicating with Anthropic API:** {str(e)}"

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator]:
        """
        Main entry point for the pipeline.
        
        Transforms Open WebUI request to Anthropic format and returns response.
        
        Args:
            user_message: Current user message (may be redundant with messages)
            model_id: The model identifier (e.g., "claude-sonnet-4-5-20250929")
            messages: Full conversation history in OpenAI format
            body: Additional request parameters (temperature, max_tokens, etc.)
        
        Returns:
            Either a complete string (non-streaming) or a generator (streaming)
        """
        # Validate API key
        if not self.valves.ANTHROPIC_API_KEY:
            return "**Error:** Anthropic API key not configured. Please set it in Admin Panel → Settings → Pipelines."
        
        # Remove model prefix if present (Open WebUI may include "anthropic/" prefix)
        if model_id.startswith("anthropic/"):
            model_id = model_id[10:]
        
        try:
            # Process messages to Anthropic format
            system_message, processed_messages = self.process_messages(messages)
            
            # Determine which features to use
            model_config = self.get_model_config(model_id)
            
            use_thinking = (
                self.valves.ENABLE_EXTENDED_THINKING and 
                model_config.get("supports_thinking", False) and
                body.get("enable_thinking", True)  # Default to enabled for supported models
            )
            
            use_caching = (
                self.valves.ENABLE_PROMPT_CACHING and
                model_config.get("supports_caching", True) and
                system_message and
                len(system_message) >= self.valves.MIN_CACHE_TOKENS
            )
            
            # Build headers with appropriate beta features
            headers = self._get_headers(use_thinking=use_thinking, use_caching=use_caching)
            
            # Build API payload
            payload = self.build_payload(
                model_id=model_id,
                messages=processed_messages,
                system_message=system_message,
                body=body,
                use_thinking=use_thinking,
                use_caching=use_caching
            )
            
            # Return streaming or complete response
            if body.get("stream", False):
                return self.stream_response(payload, headers)
            else:
                return self.get_completion(payload, headers)
        
        except ValueError as e:
            return f"**Validation Error:** {str(e)}"
        except Exception as e:
            return f"**Unexpected Error:** {str(e)}"

    def get_usage_stats(self) -> dict:
        """Returns accumulated token usage for the session."""
        return self.session_usage.copy()

    def reset_usage_stats(self):
        """Resets the token usage accumulator."""
        self.session_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "thinking_tokens": 0
        }