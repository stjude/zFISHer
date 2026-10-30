"""``zfisher-mcp``: run the server over stdio for an MCP client.

    zfisher-mcp --root /path/to/batch_root
    zfisher-mcp --root screen=/data/screen --root pilot=/data/pilot

An MCP client starts this command itself; its configuration names the command
and the arguments. Messages go over stdin and stdout; logging goes to stderr.
"""
import argparse
import logging
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zfisher-mcp", description=(
        "Answer questions about zFISHer sessions for an MCP client, over stdio. "
        "Reads the session folders under the roots and writes nothing."))
    parser.add_argument("--root", action="append", required=True, metavar="[NAME=]PATH",
                        help="A folder of zFISHer sessions the server may read; repeat for several, "
                             "each then NAME=PATH (paths in results start with NAME).")
    parser.add_argument("--mode", default="query", help="Which tools to offer: query (read-only).")
    parser.add_argument("--page-size", type=int, default=200,
                        help="The most rows or records one call returns (default 200).")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="How many folders below a root to look for sessions (default 3).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log at DEBUG to stderr.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    from .paths import Roots
    from .server import MODES, ServerConfig, build_server
    if args.mode not in MODES:
        parser.error(f"no mode '{args.mode}'; the modes are {', '.join(MODES)}")
    if args.page_size < 1 or args.max_depth < 0:
        parser.error("--page-size must be at least 1 and --max-depth at least 0")
    try:
        roots = Roots.parse(args.root)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.WARNING)
    logging.getLogger("zfisher").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    server = build_server(ServerConfig(roots=roots, mode=args.mode, page_size=args.page_size,
                                       max_depth=args.max_depth, log_level="DEBUG" if args.verbose else "WARNING"))
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
