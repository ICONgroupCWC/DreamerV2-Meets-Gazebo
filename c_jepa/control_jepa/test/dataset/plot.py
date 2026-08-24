# plot_positions.py
import sys
import os
import matplotlib.pyplot as plt

DATASET_FILE = "dataset.pt"   # <-- change to your filename if needed


def main(filename):
    try:
        import torch
    except Exception as e:
        print("ERROR: PyTorch is required to load this dataset file.")
        print("If you don't have torch installed, install it (CPU-only):")
        print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("\nFull import error:", e)
        sys.exit(1)

    if not os.path.exists(filename):
        print("ERROR: dataset file not found:", filename)
        sys.exit(1)

    print("Loading dataset with torch.load(..., map_location='cpu') -> this avoids GPU allocations")
    data = torch.load(filename, map_location="cpu")

    # Basic structure check
    if not isinstance(data, dict):
        print("WARNING: loaded object is not a dict. Inspecting type:", type(data))
        # try to handle wrapper like {"dataset": {...}}
        if hasattr(data, "__dict__"):
            print("Object has __dict__ keys:", list(data.__dict__.keys()))

    if "poses" not in data:
        print("ERROR: 'poses' key not found in loaded data. Available keys:", list(data.keys()))
        sys.exit(1)

    poses = data["poses"]

    # If poses is a tensor, convert to numpy
    import numpy as np
    if hasattr(poses, "numpy"):
        poses_np = poses.numpy()
    else:
        # assume list of [x,y]
        poses_np = np.array(poses, dtype=float)

    if poses_np.ndim != 2 or poses_np.shape[1] < 2:
        print("ERROR: poses have unexpected shape:", poses_np.shape)
        sys.exit(1)

    x = poses_np[:, 0]
    y = poses_np[:, 1]

    print("Total poses:", len(x))
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=4)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Robot Positions (from {os.path.basename(filename)})")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    fname = DATASET_FILE
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    main(fname)
