"""
optimumai.scratchpad.cli
--------------------------
Launches the local scratchpad server and opens the browser. This module stays
framework-agnostic (no click import) so the server can also be started from a
notebook or a plain script; the ``optimumai scratchpad`` command in
optimumai/cli/main.py is a thin click wrapper over ``launch()``.
"""

from __future__ import annotations

import threading
import time
import webbrowser

from .server import create_app

DEFAULT_PORT = 5057


def launch(
    concept: str = "dot_product",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """
    Start the scratchpad server.

    Args:
        concept: which concept board to open first, e.g. "dot_product" or
            "tangent_line". See optimumai.scratchpad.concepts.CONCEPTS for
            the full list.
        port: local port to bind. Nothing external is exposed — this is a
            127.0.0.1-only server by design (local-first, no cloud dependency).
        open_browser: whether to auto-open the default browser.
    """
    app = create_app()
    url = f"http://127.0.0.1:{port}/scratchpad/{concept}"

    if open_browser:
        def _open():
            time.sleep(0.75)  # give Flask a moment to bind before opening
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    print(f"optimumai scratchpad running at {url}")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=port, debug=False)
