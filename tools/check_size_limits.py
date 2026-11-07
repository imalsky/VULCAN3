#!/usr/bin/env python3
"""Check the hard size limits on source files. Run by pre-commit and CI.

The limits are the constants below.

Python: module docstring, file and function length. Python complexity is Ruff's C901.
C, C++ and CUDA: file and function length, and complexity by a scan with comments and strings removed.
It counts branches as C901 does (if, for, while, case, catch), not boolean operators.

An exception needs a one-line reason beside it: a comment holding ``size-limit-ok: <reason>`` on the line of the
function or the line above it, or in the first lines of the file for a file or module docstring limit.
Takes files as arguments, or every tracked file with none. Skips experiments/. Exits 1 when a limit is broken.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

MAX_DOCSTRING_LINES = 15
MAX_FILE_LINES = 800
# Not 80: that invited mechanical splits, and a step that reads as one ordered operation stays whole.
MAX_FUNCTION_LINES = 150
MAX_COMPLEXITY = 15
FILE_EXCEPTION_LINES = 5  # a file-wide exception must sit this close to the top, where a reader sees it

PYTHON_SUFFIXES = {".py", ".pyi"}
C_FAMILY_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}
SKIPPED_FOLDER = "experiments"  # scripts there need only format, lint and a header

EXCEPTION = re.compile(r"size-limit-ok:\s*\S")
C_NOISE = re.compile(r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'""", re.DOTALL)
C_PREPROCESSOR = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.MULTILINE)
C_BRANCH = re.compile(r"\b(?:if|for|while|case|catch)\b")
C_SIGNATURE_END = re.compile(r"\)\s*(?:(?:const|noexcept|override|final|mutable)\s*)*(?:->[^;{}()]*)?$")
C_CONTROL_WORDS = {"if", "for", "while", "switch", "catch", "sizeof", "return"}
C_WORD_BEFORE = re.compile(r"([A-Za-z_~][\w:~]*)\s*$")


def _blank(match: re.Match[str]) -> str:
    """Return the matched text with everything but newlines removed, so line numbers stay right."""
    return "\n" * match.group().count("\n")


def _has_exception(lines: list[str], line_number: int) -> bool:
    """Say whether the 1-based line, or the line above it, holds an exception with a reason."""
    return any(EXCEPTION.search(lines[i]) for i in (line_number - 1, line_number - 2) if 0 <= i < len(lines))


def _file_has_exception(lines: list[str]) -> bool:
    """Say whether the top of the file holds an exception with a reason."""
    return any(EXCEPTION.search(line) for line in lines[:FILE_EXCEPTION_LINES])


def _check_file_length(path: Path, lines: list[str]) -> list[str]:
    """Return the failure for a file over the line limit, unless it holds an exception."""
    if len(lines) > MAX_FILE_LINES and not _file_has_exception(lines):
        return [f"{path}:1: file has {len(lines)} lines, limit {MAX_FILE_LINES}"]
    return []


def check_python(path: Path, text: str) -> list[str]:
    """Check one Python file.

    Parameters
    ----------
    path : pathlib.Path
        The file, used in messages.
    text : str
        Its contents.

    Returns
    -------
    list of str
        One message per broken limit.
    """
    lines = text.splitlines()
    failures = _check_file_length(path, lines)
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [*failures, f"{path}:{error.lineno}: cannot parse: {error.msg}"]
    docstring = ast.get_docstring(tree, clean=False)
    if docstring is not None:
        count = docstring.count("\n") + 1
        if count > MAX_DOCSTRING_LINES and not _file_has_exception(lines):
            failures.append(f"{path}:1: module docstring has {count} lines, limit {MAX_DOCSTRING_LINES}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > MAX_FUNCTION_LINES and not _has_exception(lines, node.lineno):
                failures.append(f"{path}:{node.lineno}: {node.name} has {length} lines, limit {MAX_FUNCTION_LINES}")
    return failures


def _function_name(head: str) -> str | None:
    """Return the name of the function whose signature ends this text, or None if it is not a function.

    Parameters
    ----------
    head : str
        The code before an opening brace.

    Returns
    -------
    str or None
        The word before the parameter list, unless that word is a control keyword.
    """
    match = C_SIGNATURE_END.search(head)
    if match is None:
        return None
    depth = 0
    for position in range(match.start(), -1, -1):
        if head[position] == ")":
            depth += 1
        elif head[position] == "(":
            depth -= 1
            if depth == 0:
                word = C_WORD_BEFORE.search(head[:position])
                return None if word is None or word.group(1) in C_CONTROL_WORDS else word.group(1)
    return None


def c_functions(code: str) -> list[tuple[str, int, int]]:
    """Find the outermost function bodies in C-family code with comments, strings and macros removed.

    Parameters
    ----------
    code : str
        The cleaned source.

    Returns
    -------
    list of tuple
        Name, position of the name's line start and position after the closing brace, per function.
    """
    functions = []
    depth = 0
    current: tuple[str, int, int] | None = None
    for match in re.finditer(r"[{}]", code):
        if match.group() == "{":
            if current is None:
                name = _function_name(code[: match.start()])
                if name is not None:
                    start = code.rfind(name, 0, match.start())
                    current = (name, code.rfind("\n", 0, start) + 1, depth)
            depth += 1
        else:
            depth -= 1
            if current is not None and depth == current[2]:
                functions.append((current[0], current[1], match.end()))
                current = None
    return functions


def check_c_family(path: Path, text: str) -> list[str]:
    """Check one C, C++ or CUDA file.

    Parameters
    ----------
    path : pathlib.Path
        The file, used in messages.
    text : str
        Its contents.

    Returns
    -------
    list of str
        One message per broken limit.
    """
    lines = text.splitlines()
    failures = _check_file_length(path, lines)
    code = C_PREPROCESSOR.sub(_blank, C_NOISE.sub(_blank, text))
    for name, start, end in c_functions(code):
        first_line = code.count("\n", 0, start) + 1
        if _has_exception(lines, first_line):
            continue
        body = code[start:end]
        length = body.count("\n") + 1
        complexity = 1 + len(C_BRANCH.findall(body))
        if length > MAX_FUNCTION_LINES:
            failures.append(f"{path}:{first_line}: {name} has {length} lines, limit {MAX_FUNCTION_LINES}")
        if complexity > MAX_COMPLEXITY:
            failures.append(f"{path}:{first_line}: {name} has complexity {complexity}, limit {MAX_COMPLEXITY}")
    return failures


def tracked_files() -> list[Path]:
    """Return the existing files git tracks in the current directory."""
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [Path(line) for line in result.stdout.splitlines() if Path(line).is_file()]


def main(arguments: list[str]) -> int:
    """Check the named files, or every tracked file.

    Parameters
    ----------
    arguments : list of str
        File paths.

    Returns
    -------
    int
        0 when every limit holds, else 1.
    """
    paths = [Path(argument) for argument in arguments] or tracked_files()
    failures = []
    for path in paths:
        if SKIPPED_FOLDER in path.parts:
            continue
        if path.suffix in PYTHON_SUFFIXES:
            failures += check_python(path, path.read_text(encoding="utf-8"))
        elif path.suffix in C_FAMILY_SUFFIXES:
            failures += check_c_family(path, path.read_text(encoding="utf-8"))
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
