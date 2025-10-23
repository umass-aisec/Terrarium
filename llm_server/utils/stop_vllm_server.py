#!/usr/bin/env python3
"""
Script to gracefully stop a running vLLM server.

This script stops the vLLM server that was started by start_vllm_server.py.
"""

import argparse
import json
import os
import signal
import sys
import time
import requests
from typing import Dict, Any


def load_config(config_file: str = "configs/config.json") -> Dict[str, Any]:
    """Load vLLM configuration from config file."""
    default_vllm_config = {
        "host": "localhost", 
        "port": 8000
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
        return default_vllm_config


def is_server_running(host: str, port: int) -> bool:
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_server_pid() -> int:
    """Get the PID of the running server from pid file."""
    try:
        with open("logs/vllm_server.pid", "r") as pid_file:
            return int(pid_file.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def stop_server_by_pid(pid: int, timeout: int = 30) -> bool:
    """Stop server using its PID."""
    try:
        print(f"Stopping vLLM server with PID: {pid}")
        
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to terminate
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                os.kill(pid, 0)  # This doesn't kill, just checks if process exists
                time.sleep(1)
                print(".", end="", flush=True)
            except OSError:
                # Process has terminated
                print("\nvLLM server stopped gracefully")
                return True
        
        # If we reach here, graceful shutdown failed
        print(f"\nGraceful shutdown timed out after {timeout}s, force killing...")
        os.kill(pid, signal.SIGKILL)
        
        # Wait a bit more for force kill
        time.sleep(2)
        try:
            os.kill(pid, 0)
            print("Failed to stop server")
            return False
        except OSError:
            print("Server force-killed successfully")
            return True
            
    except OSError as e:
        if e.errno == 3:  # No such process
            print("Server process not found (may have already stopped)")
            return True
        else:
            print(f"Error stopping server: {e}")
            return False


def cleanup_files():
    """Clean up server-related files."""
    files_to_remove = ["logs/vllm_server.pid"]
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed {file_path}")
        except OSError as e:
            print(f"Warning: Could not remove {file_path}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stop persistent vLLM server")
    parser.add_argument("--config", default="config.json", 
                       help="Configuration file path")
    parser.add_argument("--force", action="store_true",
                       help="Force kill server immediately")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    host = config["host"]
    port = config["port"]
    
    # Check if server is running
    if not is_server_running(host, port):
        print(f"No vLLM server running at {host}:{port}")
        cleanup_files()
        sys.exit(0)
    
    # Get server PID
    pid = get_server_pid()
    if pid is None:
        print("Warning: Could not find server PID file")
        print("Server may be running but not started by start_vllm_server.py")
        print("Try manually finding and stopping the vLLM process")
        sys.exit(1)
    
    # Stop the server
    timeout = 5 if args.force else 30
    success = stop_server_by_pid(pid, timeout)
    
    if success:
        # Wait a moment and verify server is actually stopped
        time.sleep(2)
        if not is_server_running(host, port):
            print(f"Server at {host}:{port} has been stopped successfully")
            cleanup_files()
        else:
            print("Warning: Server may still be running")
    else:
        print("Failed to stop server")
        sys.exit(1)


if __name__ == "__main__":
    main()