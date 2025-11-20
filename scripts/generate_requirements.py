#!/usr/bin/env python
"""
Scan the current project for Python imports and create a requirements file.
"""
from __future__ import annotations

import argparse
import ast
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from typing import Iterable, Set

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


def iter_python_files(root: Path, ignored_dirs: Set[str]) -> Iterable[Path]:
    """Yield Python files under root while skipping ignored directories."""
    for path in root.rglob("*.py"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        yield path


def extract_imports(path: Path) -> Set[str]:
    """Return the set of top-level module names imported in the file."""
    modules: Set[str] = set()
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[warn] Could not decode {path}", file=sys.stderr)
        return modules

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        print(f"[warn] Could not parse {path}", file=sys.stderr)
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = (alias.name or "").split(".", 1)[0]
                if top_level:
                    modules.add(top_level)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                top_level = node.module.split(".", 1)[0]
                modules.add(top_level)

    return modules


def fallback_stdlib_modules() -> Set[str]:
    """Approximate the stdlib module names for Python versions < 3.10."""
    modules = set(sys.builtin_module_names)
    stdlib_path = Path(sysconfig.get_path("stdlib"))
    for path in stdlib_path.glob("*.py"):
        modules.add(path.stem)
    for directory in stdlib_path.iterdir():
        if directory.is_dir() and (directory / "__init__.py").exists():
            modules.add(directory.name)
    return modules


def discover_local_modules(root: Path) -> Set[str]:
    """Collect top-level modules that belong to the current repository."""
    modules: Set[str] = set()
    for child in root.iterdir():
        name = child.name
        if name.startswith("."):
            continue
        if child.is_dir():
            if (child / "__init__.py").exists():
                modules.add(name)
            else:
                if any(
                    subchild.suffix == ".py"
                    for subchild in child.glob("*.py")
                    if subchild.is_file()
                ):
                    modules.add(name)
        elif child.suffix == ".py":
            modules.add(child.stem)
    return modules


def modules_to_distributions(
    modules: Set[str],
    stdlib_modules: Set[str],
    local_modules: Set[str],
) -> tuple[dict[str, str], Set[str]]:
    """
    Map modules to installed distributions using importlib metadata.

    Returns:
        A tuple of (resolved_packages, unresolved_modules).
    """
    packages_map = metadata.packages_distributions()
    resolved: dict[str, str] = {}
    unresolved_third_party: Set[str] = set()

    for module in sorted(modules):
        distributions = packages_map.get(module)
        if not distributions:
            if module not in stdlib_modules and module not in local_modules:
                unresolved_third_party.add(module)
            continue

        # Pick the first distribution alphabetically for deterministic output.
        distribution = sorted(distributions)[0]
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            if module not in stdlib_modules and module not in local_modules:
                unresolved_third_party.add(module)
            continue

        resolved[distribution] = version

    return resolved, unresolved_third_party


def write_requirements(
    requirements: dict[str, str], unresolved: Set[str], destination: Path
) -> None:
    """Write package==version pairs sorted by package name."""
    lines = [f"{name}=={requirements[name]}" for name in sorted(requirements)]
    if unresolved:
        lines.append("")
        lines.append("# Packages without resolved versions (not installed?)")
        lines.extend(sorted(unresolved))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Python files for imports, map them to installed packages, "
            "and emit a requirements file."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root to scan (default: repository root).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("requirements.generated.txt"),
        help="Where to write the generated requirement pins.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Additional directory names to ignore while scanning.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra modules to force include (useful for dynamic imports).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ignored_dirs = DEFAULT_IGNORED_DIRS | set(args.ignore or [])

    root = args.root.resolve()
    if not root.exists():
        print(f"[error] Root directory {root} does not exist.", file=sys.stderr)
        return 1

    try:
        stdlib_modules = set(sys.stdlib_module_names)  # type: ignore[attr-defined]
    except AttributeError:
        stdlib_modules = fallback_stdlib_modules()

    local_modules = discover_local_modules(root)

    modules: Set[str] = set(args.include or [])

    for py_file in iter_python_files(root, ignored_dirs):
        modules |= extract_imports(py_file)

    requirements, unresolved = modules_to_distributions(
        modules, stdlib_modules, local_modules
    )
    write_requirements(requirements, unresolved, args.output.resolve())

    print(
        f"[info] Wrote {len(requirements)} packages to {args.output.resolve()}",
        file=sys.stderr,
    )
    if unresolved:
        unresolved_list = ", ".join(sorted(unresolved))
        print(
            "[warn] The following modules could not be resolved to installed "
            f"packages and were added without exact versions: {unresolved_list}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
