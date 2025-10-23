#!/usr/bin/env python3
"""
Script to check the status of the vLLM server.

This script provides detailed information about the vLLM server status,
including health, model information, and performance metrics.
"""

import argparse
import json
import os
import sys
import time
import requests
from typing import Dict, Any, Optional


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


def check_server_health(host: str, port: int) -> Dict[str, Any]:
    """Check server health and return status information."""
    base_url = f"http://{host}:{port}"
    status = {
        "running": False,
        "healthy": False,
        "reachable": False,
        "response_time": None,
        "error": None
    }
    
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=10)
        response_time = time.time() - start_time
        
        status["reachable"] = True
        status["response_time"] = response_time
        
        if response.status_code == 200:
            status["running"] = True
            status["healthy"] = True
        else:
            status["error"] = f"Health check returned status {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        status["error"] = "Connection refused - server may not be running"
    except requests.exceptions.Timeout:
        status["error"] = "Request timed out - server may be overloaded"
    except Exception as e:
        status["error"] = f"Unexpected error: {e}"
    
    return status


def get_server_info(host: str, port: int) -> Optional[Dict[str, Any]]:
    """Get detailed server information."""
    base_url = f"http://{host}:{port}"
    
    try:
        # Get model information
        models_response = requests.get(f"{base_url}/v1/models", timeout=10)
        if models_response.status_code == 200:
            return models_response.json()
        else:
            return {"error": f"Models endpoint returned {models_response.status_code}"}
    except Exception as e:
        return {"error": f"Could not get server info: {e}"}


def get_server_pid() -> Optional[int]:
    """Get the PID of the running server from pid file."""
    try:
        with open("logs/vllm_server.pid", "r") as pid_file:
            return int(pid_file.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def check_process_status(pid: int) -> Dict[str, Any]:
    """Check if the process with given PID is running."""
    status = {
        "exists": False,
        "cmdline": None,
        "error": None
    }
    
    try:
        # Check if process exists
        os.kill(pid, 0)  # This doesn't kill, just checks if process exists
        status["exists"] = True
        
        # Try to get command line (Linux-specific)
        try:
            with open(f"/proc/{pid}/cmdline", "r") as f:
                cmdline = f.read().replace('\x00', ' ').strip()
                status["cmdline"] = cmdline
        except:
            pass  # /proc might not be available or accessible
            
    except OSError as e:
        if e.errno == 3:  # No such process
            status["error"] = "Process not found"
        else:
            status["error"] = f"Cannot check process: {e}"
    
    return status


def format_response_time(response_time: float) -> str:
    """Format response time in a readable way."""
    if response_time < 0.001:
        return f"{response_time * 1000000:.0f}μs"
    elif response_time < 1:
        return f"{response_time * 1000:.1f}ms"
    else:
        return f"{response_time:.2f}s"


def print_status(config: Dict[str, Any], verbose: bool = False):
    """Print comprehensive server status."""
    host = config["host"]
    port = config["port"]
    
    print(f"vLLM Server Status Check")
    print(f"========================")
    print(f"Target: {host}:{port}")
    print()
    
    # Check server health
    health_status = check_server_health(host, port)
    
    if health_status["running"]:
        print("✅ Server Status: RUNNING")
        if health_status["response_time"]:
            print(f"⏱️  Response Time: {format_response_time(health_status['response_time'])}")
    else:
        print("❌ Server Status: NOT RUNNING")
        if health_status["error"]:
            print(f"❗ Error: {health_status['error']}")
    
    # Check PID file and process
    pid = get_server_pid()
    if pid:
        print(f"📁 PID File: Found (PID: {pid})")
        process_status = check_process_status(pid)
        
        if process_status["exists"]:
            print(f"🔍 Process: Running (PID: {pid})")
            if verbose and process_status["cmdline"]:
                print(f"   Command: {process_status['cmdline'][:100]}...")
        else:
            print(f"⚠️  Process: Not found (stale PID file)")
    else:
        print("📁 PID File: Not found")
    
    # Get server info if running
    if health_status["running"]:
        print()
        print("Server Information:")
        print("-------------------")
        
        server_info = get_server_info(host, port)
        if server_info and "error" not in server_info:
            models = server_info.get("data", [])
            if models:
                for model in models:
                    print(f"🤖 Model: {model.get('id', 'Unknown')}")
                    if verbose:
                        print(f"   Created: {model.get('created', 'Unknown')}")
                        print(f"   Owner: {model.get('owned_by', 'Unknown')}")
            else:
                print("   No models found")
        else:
            error_msg = server_info.get("error", "Unknown error") if server_info else "Could not retrieve"
            print(f"❗ Could not get model info: {error_msg}")
    
    # Check log file
    if os.path.exists("logs/vllm_server.log"):
        stat = os.stat("logs/vllm_server.log")
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = time.ctime(stat.st_mtime)
        print()
        print(f"📜 Log File: logs/vllm_server.log ({size_mb:.1f} MB, modified: {mod_time})")
        
        if verbose:
            print("   Recent log entries:")
            try:
                with open("logs/vllm_server.log", "r") as f:
                    lines = f.readlines()
                    # Show last 5 lines
                    for line in lines[-5:]:
                        print(f"   {line.rstrip()}")
            except Exception as e:
                print(f"   Could not read log: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check vLLM server status")
    parser.add_argument("--config", default="configs/config.json", 
                       help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed information")
    parser.add_argument("--json", action="store_true",
                       help="Output status as JSON")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.json:
        # JSON output for programmatic use
        health_status = check_server_health(config["host"], config["port"])
        pid = get_server_pid()
        
        status_data = {
            "host": config["host"],
            "port": config["port"],
            "health": health_status,
            "pid": pid,
            "log_file_exists": os.path.exists("logs/vllm_server.log")
        }
        
        if health_status["running"]:
            status_data["server_info"] = get_server_info(config["host"], config["port"])
        
        print(json.dumps(status_data, indent=2))
    else:
        # Human-readable output
        print_status(config, args.verbose)


if __name__ == "__main__":
    main()