#!/usr/bin/env python3
"""
Standalone script to start a persistent vLLM server.

This script starts a vLLM server that can be used by multiple runs of main.py
without restarting each time, significantly improving development iteration speed.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import requests
from typing import Dict, Any, Optional

import torch


def load_config(config_file: str = "configs/config.json") -> Dict[str, Any]:
    """Load vLLM configuration from config file."""
    default_vllm_config = {
        "model_name": "Qwen/Qwen3-4B-Thinking-2507",
        "host": "localhost", 
        "port": 8000,
        "trust_remote_code": True,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.95
    }
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            vllm_config = config.get("vllm", {})
            # Merge with defaults
            for key, value in default_vllm_config.items():
                if key not in vllm_config:
                    vllm_config[key] = value
            return vllm_config
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Using default vLLM configuration")
        return default_vllm_config


def is_server_running(host: str, port: int) -> bool:
    """Check if vLLM server is already running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def wait_for_server(host: str, port: int, timeout: int = 300) -> bool:
    """Wait for server to become ready."""
    print(f"Waiting for vLLM server to be ready at {host}:{port}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_server_running(host, port):
            print("vLLM server is ready!")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    
    print(f"\nTimeout: Server did not start within {timeout} seconds")
    return False


def start_server(config: Dict[str, Any], background: bool = True) -> Optional[subprocess.Popen]:
    """Start the vLLM server."""
    host = config["host"]
    port = config["port"]
    model_name = config["model_name"]
    torch.cuda.empty_cache()
    # Check if server is already running
    if is_server_running(host, port):
        print(f"vLLM server is already running at {host}:{port}")
        return None
    
    print(f"Starting vLLM server with model: {model_name}")
    print(f"Server will be available at: http://{host}:{port}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port)
    ]
    
    if config.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")
    
    # Add memory and model length parameters
    if "max_model_len" in config:
        cmd.extend(["--max-model-len", str(config["max_model_len"])])
    
    if "gpu_memory_utilization" in config:
        cmd.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])
    
    if "tensor_parallel_size" in config:
        cmd.extend(["--tensor-parallel-size", str(config["tensor_parallel_size"])])
    
    # Add any additional arguments from config
    if "additional_args" in config:
        cmd.extend(config["additional_args"])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        if background:
            # Start server in background
            with open("logs/vllm_server.log", "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy()
                )
            print(f"vLLM server started in background with PID: {process.pid}")
            print(f"Logs are being written to: logs/vllm_server.log")
            
            # Save PID for later cleanup
            with open("logs/vllm_server.pid", "w") as pid_file:
                pid_file.write(str(process.pid))
            
            return process
        else:
            # Start server in foreground (for debugging)
            print("Starting server in foreground mode (use Ctrl+C to stop)...")
            process = subprocess.Popen(cmd, env=os.environ.copy())
            return process
            
    except Exception as e:
        print(f"Failed to start vLLM server: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start persistent vLLM server")
    parser.add_argument("--config", default="configs/config.json", 
                       help="Configuration file path")
    parser.add_argument("--foreground", action="store_true",
                       help="Run server in foreground (default: background)")
    parser.add_argument("--no-wait", action="store_true", 
                       help="Don't wait for server to be ready")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start server
    process = start_server(config, background=not args.foreground)
    
    if process is None:
        # Server was already running
        sys.exit(0)
    
    if args.foreground:
        # Wait for process in foreground mode
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing server...")
                process.kill()
                process.wait()
    else:
        # Background mode - wait for server to be ready
        if not args.no_wait:
            if wait_for_server(config["host"], config["port"]):
                print(f"\nvLLM server is ready! You can now run main.py")
                print(f"To stop the server, run: python stop_vllm_server.py")
                print(f"To check status, run: python check_vllm_server.py")
            else:
                print("Server may still be starting. Check vllm_server.log for details.")
                print("Use check_vllm_server.py to monitor status.")


if __name__ == "__main__":
    main()