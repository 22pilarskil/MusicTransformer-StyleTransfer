# Re-importing torch after the reset
import torch

# Function to calculate the average pairwise distance between pairs
def compute_average_pairwise_distances(y_melody, y_harmony, y_combined):
    # Calculate pairwise distances between y_harmony & y_combined
    distances_harmony_combined = torch.norm(y_harmony - y_combined, p=2, dim=1)
    # Calculate pairwise distances between y_melody & y_combined
    distances_melody_combined = torch.norm(y_melody - y_combined, p=2, dim=1)
    # Calculate pairwise distance between y_melody & y_harmony
    distances_melody_harmony = torch.norm(y_melody - y_harmony, p=2, dim=1)

    # Calculate the average distance for each pair
    average_distance_harmony_combined = torch.mean(distances_harmony_combined).item()
    average_distance_melody_combined = torch.mean(distances_melody_combined).item()
    average_distance_melody_harmony = torch.mean(distances_melody_harmony).item()

    return average_distance_melody_harmony, average_distance_melody_combined, average_distance_harmony_combined
# Assuming the shape of y_melody, y_harmony, and y_combined tensors as (batch_size, 128)
# Here, I'll create dummy tensors to simulate the computation.
# You should replace these with your actual tensors.

batch_size = 5  # Example batch size
# Example tensors
y_melody = torch.rand(batch_size, 128) + 0.5
y_harmony = torch.rand(batch_size, 128) - 0.5
y_combined = torch.rand(batch_size, 128)

# Compute the average distances

print(compute_average_pairwise_distances(y_melody, y_harmony, y_combined))
