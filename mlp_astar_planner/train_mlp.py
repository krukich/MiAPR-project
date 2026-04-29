#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ament_index_python.packages import get_package_share_directory
from mlp_astar_planner.mlp_model import OccupancyMLP


def get_project_data_dir():
    workspace_dir = os.path.abspath(
        os.path.join(
            get_package_share_directory("mlp_astar_planner"),
            "..",
            "..",
            ".."
        )
    )

    data_dir = os.path.join(
        workspace_dir,
        "src",
        "mlp_astar_planner",
        "data"
    )

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def main():
    data_dir = get_project_data_dir()

    data_path = os.path.join(data_dir, "dataset.npz")
    model_path = os.path.join(data_dir, "mlp_model.pth")

    data = np.load(data_path)
    X_np = data["X"]
    y_np = data["y"]

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.float32).to(device)

    model = OccupancyMLP().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 500000

    for epoch in range(epochs):
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            accuracy = (pred == y).float().mean()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, "
                f"loss = {loss.item():.4f}, "
                f"acc = {accuracy.item():.4f}"
            )

    torch.save(model.state_dict(), model_path)

    print("Training finished")
    print(f"Dataset loaded from: {data_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()