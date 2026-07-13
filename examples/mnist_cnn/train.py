#!/usr/bin/env python3
"""Train a tiny MNIST CNN (CPU-friendly)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 5 * 5, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--ckpt", type=Path, default=Path("mnist.pt"))
    args = p.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(str(args.data), train=True, download=True, transform=transform)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"epoch={epoch + 1} loss={total / max(len(loader), 1):.4f}")

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt)
    print(f"wrote {args.ckpt.resolve()}")


if __name__ == "__main__":
    main()
