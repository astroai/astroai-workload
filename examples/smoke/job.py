#!/usr/bin/env python3
"""Tiny Ray Jobs smoke — no ML deps."""

from __future__ import annotations

import os
import socket


def main() -> None:
    print("astroai-workload smoke ok")
    print(f"host={socket.gethostname()}")
    print(f"cwd={os.getcwd()}")


if __name__ == "__main__":
    main()
