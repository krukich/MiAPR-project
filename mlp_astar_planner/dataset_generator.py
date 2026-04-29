#!/usr/bin/env python3

import os
import yaml
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory


def load_map_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        info = yaml.safe_load(f)

    image_path = info["image"]

    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Cannot load map image: {image_path}")

    negate = info.get("negate", 0)
    occupied_thresh = info.get("occupied_thresh", 0.65)
    free_thresh = info.get("free_thresh", 0.196)

    if negate == 0:
        occ_prob = (255.0 - img.astype(np.float32)) / 255.0
    else:
        occ_prob = img.astype(np.float32) / 255.0

    occupancy = np.full(img.shape, -1.0, dtype=np.float32)
    occupancy[occ_prob > occupied_thresh] = 1.0
    occupancy[occ_prob < free_thresh] = 0.0

    return occupancy


def generate_dataset(occupancy):
    height, width = occupancy.shape

    X = []
    y = []

    for row in range(height):
        for col in range(width):
            label = occupancy[row, col]

            if label < 0:
                continue

            x_norm = col / (width - 1)
            y_norm = row / (height - 1)

            X.append([x_norm, y_norm])
            y.append([label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def main():
    pkg_share = get_package_share_directory("mlp_astar_planner")
    yaml_path = os.path.join(pkg_share, "maps", "map.yaml")

    occupancy = load_map_from_yaml(yaml_path)
    X, y = generate_dataset(occupancy)

    workspace_dir = os.path.abspath(
        os.path.join(get_package_share_directory("mlp_astar_planner"), "..", "..", "..")
    )

    package_src_dir = os.path.join(
        workspace_dir, "src", "mlp_astar_planner"
    )

    data_dir = os.path.join(package_src_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, "dataset.npz")
    np.savez(save_path, X=X, y=y, occupancy=occupancy)

    print("Dataset generated")
    print(f"Map shape: {occupancy.shape}")
    print(f"Samples: {len(X)}")
    print(f"Free: {np.sum(y == 0.0)}")
    print(f"Occupied: {np.sum(y == 1.0)}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()