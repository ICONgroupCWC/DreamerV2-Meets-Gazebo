import torch

# Load dataset
data = torch.load("dataset.pt")

poses = torch.tensor(data["poses"], dtype=torch.float32)
model_states = torch.tensor(data["model_states"], dtype=torch.float32)

print("=== DATASET STATS ===")
print("Total samples:", len(poses))
print("Pose tensor shape:", poses.shape)            # Expect: [N, 2]
print("Model state tensor shape:", model_states.shape)  # Expect: [N, 1324]

# Check alignment
if len(poses) != len(model_states):
    print("⚠ WARNING: Length mismatch!")
else:
    print("✓ Length OK: poses and model_states are aligned")

# Print first sample
print("\nFirst pose sample:", poses[0].tolist())
print("First model_state length:", model_states[0].shape[0])
print("First model_state mean:", model_states[0].mean().item())
