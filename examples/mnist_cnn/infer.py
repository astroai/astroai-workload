#!/usr/bin/env python3
"""Score a few MNIST test images with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import Net


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("mnist.pt"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--batches", type=int, default=3)
    args = p.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test = datasets.MNIST(str(args.data), train=False, download=True, transform=transform)
    loader = DataLoader(test, batch_size=64, shuffle=False)
    model = Net()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
            if i + 1 >= args.batches:
                break
    print(f"accuracy≈{correct / max(total, 1):.3f} on {total} samples")


if __name__ == "__main__":
    main()
