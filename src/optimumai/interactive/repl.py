"""An interactive REPL for OptimumAI.

    optimumai repl

Type any ``optimumai`` subcommand without the ``optimumai`` prefix — e.g.
``learn attention`` or ``softmax [2,1,0.1]``. Session variables (``set level
engineer``, ``set temperature 0.5``) are auto-applied to commands that accept
them. Uses ``prompt_toolkit`` (arrow keys, history, tab-complete) when
installed, and degrades to plain ``input()`` otherwise.
"""

from __future__ import annotations

import shlex

_BANNER = """OptimumAI REPL — type a command (e.g. 'learn attention', 'softmax [2,1,0.1]').
Commands: help · course · set <k> <v> · vars · history · clear · exit
"""

# session var -> (command names it applies to, CLI flag)
_VAR_FLAGS = {
    "level": (None, "--level"),          # None = applies to any command that accepts it
    "temperature": ({"softmax"}, "--temperature"),
}


def _command_names() -> list[str]:
    from optimumai.cli.main import cli

    return sorted(cli.commands)


def _apply_vars(args: list[str], variables: dict[str, str]) -> list[str]:
    """Inject session vars as flags when the command supports them and they're unset."""
    if not args:
        return args
    cmd = args[0]
    out = list(args)
    for name, value in variables.items():
        cmds, flag = _VAR_FLAGS.get(name, (None, f"--{name}"))
        if cmds is not None and cmd not in cmds:
            continue
        if flag in out:
            continue
        out += [flag, value]
    return out


def _dispatch(line: str, variables: dict[str, str], history: list[str]) -> bool:
    """Run one REPL line. Returns False to signal exit."""
    import click

    from optimumai.cli.main import cli

    line = line.strip()
    if not line:
        return True
    history.append(line)
    parts = shlex.split(line)
    head = parts[0].lower()

    if head in {"exit", "quit", "q"}:
        return False
    if head == "help":
        print(_BANNER)
        print("Available commands:", ", ".join(_command_names()))
        return True
    if head == "history":
        for i, item in enumerate(history[:-1], 1):
            print(f"{i:>3}  {item}")
        return True
    if head == "vars":
        print(variables or "(no session variables set)")
        return True
    if head == "set" and len(parts) >= 3:
        variables[parts[1]] = parts[2]
        print(f"set {parts[1]} = {parts[2]}")
        return True
    if head == "unset" and len(parts) >= 2:
        variables.pop(parts[1], None)
        return True
    if head == "clear":
        click.clear()
        return True

    args = _apply_vars(parts, variables)
    try:
        cli.main(args=args, prog_name="optimumai", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive on any error
        print(f"error: {exc}")
    return True


def run_repl() -> None:
    """Start the interactive read-eval-print loop."""
    print(_BANNER)
    variables: dict[str, str] = {}
    history: list[str] = []

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter

        completer = WordCompleter(
            [*_command_names(), "set", "vars", "history", "help", "exit", "clear"],
            ignore_case=True,
        )
        session: PromptSession = PromptSession(completer=completer)

        def read() -> str:
            return session.prompt("optimumai ➜ ")
    except ImportError:
        def read() -> str:
            return input("optimumai ➜ ")

    while True:
        try:
            line = read()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not _dispatch(line, variables, history):
            break
    print("bye 👋")
