#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UFM_ROOT = ROOT / "external" / "UFM"
UNICEPTION_ROOT = UFM_ROOT / "UniCeption"

# Mirrors the current install_requires declared by UniCeption.
UNICEPTION_DEPS = [
    "numpy",
    "torch",
    "torchvision",
    "torchaudio",
    "timm",
    "black",
    "jaxtyping",
    "matplotlib<3.11",
    "Pillow",
    "scikit-learn",
    "einops",
    "rerun-sdk",
    "pre-commit",
    "minio",
    "pytest",
    "isort",
]

# Extra dependencies declared by UFM on top of UniCeption.
UFM_EXTRA_DEPS = [
    "opencv-python",
    "flow_vis",
    "huggingface_hub",
    "gradio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install GeCo dependencies sequentially with clear pip progress output."
    )
    parser.add_argument(
        "--with-guidance",
        action="store_true",
        help="Also install the optional dependencies for the test time guidance experiment.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print()


def run_command(command: list[str], *, cwd: Path | None = None, label: str) -> None:
    print_section(label)
    print("Commande :", " ".join(command))
    print()
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def install_packages(packages: list[str], *, section_name: str) -> None:
    total = len(packages)
    for index, package in enumerate(packages, start=1):
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--progress-bar",
                "on",
                package,
            ],
            label=f"{section_name} [{index}/{total}] - {package}",
        )


def install_requirements_file(requirements_path: Path, *, section_name: str) -> None:
    packages = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    install_packages(packages, section_name=section_name)


def install_editable(path: Path, *, section_name: str) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "on",
            "-e",
            ".",
            "--no-deps",
        ],
        cwd=path,
        label=section_name,
    )


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def main() -> None:
    args = parse_args()

    core_packages = dedupe(
        UNICEPTION_DEPS
        + UFM_EXTRA_DEPS
        + [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    )

    print_section("GeCo dependency installer")
    print(f"Python       : {sys.executable}")
    print(f"Workspace    : {ROOT}")
    print(f"With guidance: {args.with_guidance}")
    print()

    install_packages(core_packages, section_name="Core dependencies")
    install_editable(UNICEPTION_ROOT, section_name="Editable install - UniCeption")
    install_editable(UFM_ROOT, section_name="Editable install - UFM")

    if args.with_guidance:
        install_requirements_file(
            ROOT / "requirements_guidance.txt",
            section_name="Guidance dependencies",
        )

    print_section("Installation finished")
    print("Toutes les dependances ont ete installees de maniere sequentielle.")
    print()


if __name__ == "__main__":
    main()
