#!/bin/bash
cd /Users/alex/python/mlx-project
export PATH="/Users/alex/.local/bin:$PATH"
exec uv run python server.py --host 0.0.0.0 --port 10300 --debug