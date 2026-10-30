"""An MCP server over zFISHer's results.

The server answers questions about finished zFISHer sessions (what a session
holds, what is out of date, how it was made, its quality-control numbers, its
report tables and neighbour distances) for any client that speaks the Model
Context Protocol. Every tool calls ``zfisher.core.api`` and returns its typed
result; the server holds no analysis logic of its own and imports nothing of
zFISHer's GUI.
"""
__version__ = "0.1.0"
