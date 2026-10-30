"""``zfisher-mcp``: run the server over stdio for an MCP client, or over HTTP.

    zfisher-mcp --root /path/to/batch_root
    zfisher-mcp --root screen=/data/screen --root pilot=/data/pilot
    zfisher-mcp --root /scratch/tuning_copies --mode operate --actor agent:assistant

An MCP client starts this command itself; its configuration names the command
and the arguments. Messages go over stdin and stdout; logging goes to stderr.
With ``--http [HOST:]PORT`` the server listens on that address instead (this
machine only, unless ``--allow-remote``), and every request must carry its
bearer token (``--token``, the ``ZFISHER_MCP_TOKEN`` variable, or one made up
and printed at start).
"""
import argparse
import logging
import os
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="zfisher-mcp", description=(
        "Answer questions about zFISHer sessions for an MCP client, over stdio. In query mode "
        "(the default) it reads the session folders under the roots and writes nothing."))
    parser.add_argument("--root", action="append", required=True, metavar="[NAME=]PATH",
                        help="A folder of zFISHer sessions the server may read; repeat for several, "
                             "each then NAME=PATH (paths in results start with NAME).")
    parser.add_argument("--mode", default="query", help=(
        "Which tools to offer: query (read-only, the default), triage (adds flag_batch and propose) "
        "or operate (adds changing settings and running stages and batches)."))
    parser.add_argument("--actor", default="agent", help=(
        "Who the decisions log names for changes made through the server (default: agent; e.g. "
        "agent:<model>)."))
    parser.add_argument("--max-workers", type=int, default=1,
                        help="The most datasets run_batch runs at once (default 1).")
    parser.add_argument("--page-size", type=int, default=200,
                        help="The most rows or records one call returns (default 200).")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="How many folders below a root to look for sessions (default 3).")
    parser.add_argument("--http", metavar="[HOST:]PORT", help=(
        "Serve over Streamable HTTP at http://HOST:PORT/mcp instead of stdio (HOST defaults to 127.0.0.1)."))
    parser.add_argument("--token", help="The bearer token HTTP clients must send (default: ZFISHER_MCP_TOKEN, "
                                        "else one made up and printed).")
    parser.add_argument("--allow-remote", action="store_true", help=(
        "Allow --http on an address other machines can reach. Put a TLS proxy in front: the token "
        "travels in plain text otherwise."))
    parser.add_argument("-v", "--verbose", action="store_true", help="Log at DEBUG to stderr.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    from .paths import Roots
    from .server import MODES, ServerConfig, build_server
    if args.mode not in MODES:
        parser.error(f"no mode '{args.mode}'; the modes are {', '.join(MODES)}")
    if args.page_size < 1 or args.max_depth < 0 or args.max_workers < 1:
        parser.error("--page-size and --max-workers must be at least 1 and --max-depth at least 0")
    if not args.actor.strip() or args.actor.strip().lower() in ("human", "default"):
        parser.error("--actor names the agent; 'human' and 'default' mean something else in the decisions log")
    try:
        roots = Roots.parse(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.http:
        from .httpserver import is_loopback
        host, _, port = args.http.rpartition(":")
        host = host.strip("[]") or "127.0.0.1"
        if not port.isdigit():
            parser.error("--http takes [HOST:]PORT, e.g. 127.0.0.1:8765")
        if not is_loopback(host) and not args.allow_remote:
            parser.error(f"--http {host} can be reached from other machines; add --allow-remote if that is meant")

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.WARNING)
    logging.getLogger("zfisher").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    server = build_server(ServerConfig(roots=roots, mode=args.mode, page_size=args.page_size,
                                       max_depth=args.max_depth, log_level="DEBUG" if args.verbose else "WARNING",
                                       actor=args.actor.strip(), max_workers=args.max_workers))
    if not args.http:
        server.run("stdio")
        return 0
    from .httpserver import HttpServer
    token = args.token or os.environ.get("ZFISHER_MCP_TOKEN") or None
    http = HttpServer(server, host=host, port=int(port), token=token)
    print(f"zfisher-mcp: serving {args.mode} mode at {http.url}", file=sys.stderr)
    if token is None:
        print(f"zfisher-mcp: bearer token {http.token}", file=sys.stderr)
    http.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
