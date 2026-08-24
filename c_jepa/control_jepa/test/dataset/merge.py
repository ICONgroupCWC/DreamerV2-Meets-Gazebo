import torch

# List all your dataset files here
files = [
    "1.pt",
    "dataset.pt",
]

merged_poses = []
merged_states = []

for f in files:
    print("Loading:", f)
    data = torch.load(f)

    merged_poses.extend(data["poses"])
    merged_states.extend(data["model_states"])

print("\n=== MERGE COMPLETE ===")
print("Total samples:", len(merged_poses))
print("Total model states:", len(merged_states))

# Save final combined dataset
torch.save(
    {"poses": merged_poses, "model_states": merged_states},
    "dataset.pt"
)

print("\nSaved merged dataset to merged_gazebo_modelstate_dataset.pt")
