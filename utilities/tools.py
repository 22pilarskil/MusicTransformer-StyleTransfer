import torch


def compute_triplet_distances(embeddings, labels, margin, return_all=False):
    triplet_indices = create_triplet_mask(labels)
    anchor_embeddings = embeddings[triplet_indices[:, 0]]
    positive_embeddings = embeddings[triplet_indices[:, 1]]
    negative_embeddings = embeddings[triplet_indices[:, 2]]

    pos_dist = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings, p=2)
    neg_dist = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings, p=2)

    semi_hard_triplet_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + margin)
    hard_triplet_mask = (pos_dist - neg_dist + margin > 0)

    zero_loss_triplets = ~(semi_hard_triplet_mask | (pos_dist - neg_dist + margin > 0))
    zero_loss_count = zero_loss_triplets.sum().item()
    print("Zero Loss Triplets:", zero_loss_count)

    if return_all:
        combined_mask = semi_hard_triplet_mask | hard_triplet_mask
        valid_indices = triplet_indices[combined_mask]
    else:
        valid_indices = triplet_indices[semi_hard_triplet_mask] if semi_hard_triplet_mask.any() else triplet_indices[hard_triplet_mask]

    print(f"Computing {len(valid_indices)} / {len(triplet_indices)}")
    return valid_indices


def create_triplet_mask(labels):
    # Initialize empty mask
    num_labels = len(labels)
    valid_triplets = torch.zeros((num_labels, num_labels, num_labels), dtype=torch.bool)
    for i in range(num_labels):
        for j in range(num_labels):
            if i == j:
                continue
            for k in range(num_labels):
                if k == i:
                    continue
                # Mark as valid if i and j share at least one label, but k has no overlap with i
                if not set(labels[i].numpy()).isdisjoint(set(labels[j].numpy())) and set(labels[i].numpy()).isdisjoint(set(labels[k].numpy())):
                    valid_triplets[i, j, k] = True
    valid_triplets = valid_triplets.nonzero(as_tuple=False)
    return valid_triplets



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
