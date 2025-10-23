"""
vLLM client implementation for local model serving.
"""

import requests
import time
from typing import Dict, Any, Optional, List
import subprocess
import os
import threading
from llm_server.clients.abstract_client import AbstractClient


class VLLMClient(AbstractClient):
    """
    Client for using vLLM server for LLM agent inference.
    
    Handles starting the vLLM server with Qwen3-4B-Thinking model and
    managing agent communications with thinking mode enabled.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
                 host: str = "localhost", port: int = 8000,
                 auto_start_server: bool = True, connect_only: bool = False,
                 connection_timeout: int = 30, max_retries: int = 3):
        """
        Initialize the vLLM client.
        
        Args:
            model_name: HuggingFace model name to use
            host: Server host address
            port: Server port
            auto_start_server: Whether to automatically start the server if not running
            connect_only: If True, only try to connect to existing server, don't start new one
            connection_timeout: Timeout for initial connection attempts
            max_retries: Maximum connection retry attempts
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        self.connect_only = connect_only
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        
        # Standard vLLM server initialization
        # Try connecting to existing server first
        if self._try_connect_existing():
            print(f"Connected to existing vLLM server at {self.base_url}")
            return
        
        # If connect_only mode and no server found, raise error
        if connect_only:
            raise ConnectionError(f"No vLLM server found at {self.base_url} and connect_only=True")
        
        # Start server if auto_start_server is enabled
        if auto_start_server:
            self.start_server()
            # Wait for server to be ready
            self._wait_for_server()
        else:
            raise ConnectionError(f"No vLLM server found at {self.base_url} and auto_start_server=False")

    def set_meta_context(self, agent_name: str, phase: str, iteration: Optional[int] = None, round_num: Optional[int] = None):
        """
        Set the current agent context for tool call logging (no-op for vLLM client).

        Args:
            agent_name: Name of the current agent
            phase: Current phase (planning, execution, etc.)
            iteration: Current iteration number
            round_num: Current round number
        """
        # No-op for vLLM client - this is used for tool call logging in OpenAI client
        pass

    def start_server(self):
        """
        Start the vLLM server with the specified model.
        
        Args:
            timeout: Maximum time to wait for server startup
        """
        print(f"Starting vLLM server with model: {self.model_name}")
        
        # Check if server is already running
        if self._is_server_running():
            print("vLLM server is already running")
            return
        
        # Start the server
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--trust-remote-code"
        ]
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            print(f"vLLM server started with PID: {self.server_process.pid}")
            
            # Start logging thread to capture server output; used only for DEBUGGING
            # self._start_logging_thread()
            
        except Exception as e:
            print(f"Failed to start vLLM server: {e}")
            raise
    
    def _start_logging_thread(self):
        """Start a background thread to log server output."""
        def log_output():
            if not self.server_process or not self.server_process.stdout or not self.server_process.stderr:
                return
                
            # Log stdout
            for line in iter(self.server_process.stdout.readline, b''):
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line:
                        print(f"[vLLM] {decoded_line}")
            
            # Log stderr
            for line in iter(self.server_process.stderr.readline, b''):
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line:
                        print(f"[vLLM ERROR] {decoded_line}")
        
        # Start logging thread
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
    
    def stop_server(self):
        """Stop the vLLM server if it was started by this client."""
        if self.server_process:
            print("Stopping vLLM server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing vLLM server...")
                self.server_process.kill()
                self.server_process.wait()
            
            self.server_process = None
            print("vLLM server stopped")
    
    def _is_server_running(self) -> bool:
        """Check if the vLLM server is running and responding."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _try_connect_existing(self) -> bool:
        """Try to connect to an existing vLLM server with retries."""
        print(f"Attempting to connect to existing vLLM server at {self.base_url}...")
        
        for attempt in range(self.max_retries):
            if self._is_server_running():
                # Verify the server is running the expected model
                if self._verify_server_model():
                    return True
                else:
                    print(f"Server is running but with different model")
                    return False
            
            if attempt < self.max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(2)
        
        print(f"Could not connect to existing server after {self.max_retries} attempts")
        return False
    
    def _verify_server_model(self) -> bool:
        """Verify that the server is running the expected model."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("data", [])
                if models:
                    server_model = models[0].get("id", "")
                    # Check if it's the same model (might have slightly different format)
                    expected_model_name = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
                    server_model_name = server_model.split("/")[-1] if "/" in server_model else server_model
                    
                    if expected_model_name.lower() in server_model_name.lower():
                        print(f"Server model verified: {server_model}")
                        return True
                    else:
                        print(f"Model mismatch - Expected: {self.model_name}, Server: {server_model}")
                        return False
            return False
        except Exception as e:
            print(f"Could not verify server model: {e}")
            return True  # Assume it's OK if we can't verify
    
    def _wait_for_server(self, timeout: int = 600):
        """
        Wait for the vLLM server to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        print("Waiting for vLLM server to be ready...")
        
        while time.time() - start_time < timeout:
            if self._is_server_running():
                print("vLLM server is ready!")
                return
            time.sleep(2)
        
        raise TimeoutError(f"vLLM server did not start within {timeout} seconds")
    
    def generate_response(
        self,
        input: List[Any],
        params: dict[str, Any],
    ) -> tuple[Any, List[Any], str]:
        """
        TODO: CHECK THIS GENERATED WITH CLAUDE
        Generate a response using vLLM server (AbstractClient interface).

        Args:
            input: List of message dictionaries (OpenAI chat format)
            params: Generation parameters including:
                - model: Model name (optional, uses self.model_name if not provided)
                - max_tokens or max_completion_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (response_dict, updated_messages, response_text)
        """
        # Extract parameters
        max_tokens = params.get("max_tokens") or params.get("max_completion_tokens", 1000)
        temperature = params.get("temperature", 0.7)
        enable_thinking = params.get("enable_thinking", True)

        # Use vLLM's native chat completion endpoint
        payload = {
            "model": self.model_name,
            "messages": input,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 1
        }

        # Add thinking mode if supported by model
        if enable_thinking and "thinking" in self.model_name.lower():
            payload["extra_body"] = {"enable_thinking": True}

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")

            result = response.json()

            # Extract response content
            content = result["choices"][0]["message"]["content"]

            # Parse thinking vs. response using token-based approach for Qwen3 models
            thinking_content = ""
            response_content = content

            if enable_thinking and "thinking" in self.model_name.lower():
                # Get logprobs data to find </think> token
                logprobs_data = result["choices"][0].get("logprobs")
                if logprobs_data and logprobs_data.get("content"):
                    tokens = logprobs_data["content"]

                    # Find the </think> token position
                    think_end_index = None
                    for i, token_data in enumerate(tokens):
                        if token_data.get("token") == "</think>":
                            think_end_index = i
                            break

                    if think_end_index is not None:
                        # Extract thinking content (everything before </think>)
                        thinking_tokens = tokens[:think_end_index]
                        thinking_content = "".join([t.get("token", "") for t in thinking_tokens]).strip()

                        # Extract response content (everything after </think>)
                        response_tokens = tokens[think_end_index + 1:]
                        response_content = "".join([t.get("token", "") for t in response_tokens]).strip()

            # Build assistant message to add to context
            assistant_message = {"role": "assistant", "content": content}

            # Update context with new message
            updated_context = input.copy() if isinstance(input, list) else [input]
            updated_context.append(assistant_message)

            return result, updated_context, response_content

        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    def generate_response_legacy(self, system_prompt: str, user_prompt: str,
                         max_tokens: int = 1000, temperature: float = 0.7,
                         enable_thinking: bool = True) -> Dict[str, Any]:
        """
        Generate a response from the LLM with optional thinking mode.
        
        Args:
            system_prompt: System prompt for the agent
            user_prompt: User prompt with context and query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            enable_thinking: Whether to enable thinking mode
            
        Returns:
            Dictionary containing response and metadata
        """
        # Prepare the request payload
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 1
        }
        
        # Add thinking mode if supported by model
        if enable_thinking and "thinking" in self.model_name.lower():
            payload["extra_body"] = {"enable_thinking": True}
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract response content
            content = result["choices"][0]["message"]["content"]
            
            # Parse thinking vs. response using token-based approach for Qwen3 models
            thinking_content = ""
            response_content = content
            
            if enable_thinking and "thinking" in self.model_name.lower():
                # Get logprobs data to find </think> token (ID 151668)
                logprobs_data = result["choices"][0].get("logprobs")
                if logprobs_data and logprobs_data.get("content"):
                    tokens = logprobs_data["content"]
                    
                    # Find the </think> token position
                    think_end_index = None
                    for i, token_data in enumerate(tokens):
                        if token_data.get("token") == "</think>":
                            think_end_index = i
                            break
                    
                    if think_end_index is not None:
                        # Split content at the </think> token position
                        # Extract thinking content (everything before </think>)
                        thinking_tokens = tokens[:think_end_index]
                        thinking_content = "".join([t.get("token", "") for t in thinking_tokens]).strip()
                        
                        # Extract response content (everything after </think>)
                        response_tokens = tokens[think_end_index + 1:]
                        response_content = "".join([t.get("token", "") for t in response_tokens]).strip()
                    else:
                        # No </think> token found, treat entire content as response
                        thinking_content = ""
                        response_content = content
            
            return {
                "response": response_content,
                "thinking": thinking_content,
                "full_content": content,
                "usage": result.get("usage", {}),
                "model": result.get("model", self.model_name)
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            raise
    
    def generate_agent_response(self, agent_name: str, agent_context: Dict[str, Any],
                              blackboard_context: Dict[str, str], system_prompt: str,
                              user_prompt: Optional[str] = None,
                              mcp_tools: Optional[List] = None, available_tools: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate a response for a specific agent with full context.

        Args:
            agent_name: Name of the agent
            agent_context: Agent's private context (budget, inventory, utilities)
            blackboard_context: Dictionary mapping blackboard_id -> context_summary for agent's blackboards
            system_prompt: Base system prompt for the agent
            user_prompt: Optional user prompt (if not provided, will generate using default logic)
            mcp_tools: Optional MCP tools (passed for compatibility, not used in vLLM)
            available_tools: Optional available tools list (passed for compatibility, not used in vLLM)

        Returns:
            Dictionary containing agent's response and thinking
        """
        # User prompt must be provided by the environment
        if user_prompt is None:
            raise ValueError("user_prompt must be provided by the calling environment")
        
        return self.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_thinking=True
        )
    
    def __del__(self):
        """Cleanup when the client is destroyed."""
        self.stop_server()