# vLLM Server Management
## Quick Start
### Option 1: Persistent Mode

1. **Start the server once:**
   ```bash
   python server/utils/start_vllm_server.py
   ```

2. **Run your experiments multiple times:**
   ```bash
   python main.py  # Connects to existing server
   # ... repeat as needed
   ```

3. **Ensure to stop the server when done:**
   ```bash
   python server/utils/stop_vllm_server.py
   ```

### Option 2: Auto-start-stop Mode

Set `prefer_existing_server: false` in `configs/config.json`, then:
```bash
python main.py  # Will start and stop server automatically
```

## Server Management Commands

### Start Server
```bash
# Start in background (default)
python vllm_server/start_vllm_server.py

# Start in foreground (for debugging)
python vllm_server/start_vllm_server.py --foreground

# Start without waiting for ready state
python vllm_server/start_vllm_server.py --no-wait

# Use custom config
python vllm_server/start_vllm_server.py --config configs/my_config.json
```

### Check Server Status
```bash
# Basic status check
python vllm_server/check_vllm_server.py

# Verbose output with logs
python vllm_server/check_vllm_server.py --verbose

# JSON output for scripts
python vllm_server/check_vllm_server.py --json
```

### Stop Server
```bash
# Shutdown
python vllm_server/stop_vllm_server.py

# Force kill
python vllm_server/stop_vllm_server.py --force
```

## Configuration

The vLLM server configuration is in `config.json` under the `vllm` section:

```json
{
  "vllm": {
    "model_name": "Qwen/Qwen3-4B-Thinking-2507",
    "host": "localhost",
    "port": 8000,
    "prefer_existing_server": true,
    "auto_start_server": true,
    "trust_remote_code": true,
    "connection_timeout": 30,
    "max_retries": 3,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.95,
    "additional_args": []
  }
}
```

### Configuration Options

- **`prefer_existing_server`**: Try connecting to existing server first (recommended: `true`)
- **`auto_start_server`**: Start new server if none found (recommended: `true`)
- **`model_name`**: HuggingFace model to load
- **`host`/`port`**: Server address (Don't need to change)
- **`max_model_len`**: Maximum context length (default: 8192)
- **`gpu_memory_utilization`**: GPU memory utilization fraction (default: 0.95)
- **`additional_args`**: Extra command-line arguments for vLLM server

## Troubleshooting

### Server Won't Stop
```bash
# Force kill
python vllm_server/stop_vllm_server.py --force

# Manual cleanup
ps aux | grep vllm
kill -9 <PID>
rm -f vllm_server.pid
```