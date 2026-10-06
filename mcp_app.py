"""Entry point for the MCP server: `openbb-mcp --app mcp_app.py`.

openbb-mcp loads this file by path, so make the package importable first.
Every /api route in app/main.py becomes an MCP tool named by its operation_id.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app  # noqa: E402,F401
