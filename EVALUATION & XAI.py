def recall_at_k(indices, query_labels, gallery_labels, k=1):
    correct = 0

    for i in range(len(query_labels)):
        top_indices = indices[i][:k].tolist()
        top_labels = [gallery_labels[idx] for idx in top_indices]

        if query_labels[i] in top_labels:
            correct += 1

    return correct / len(query_labels)


r1 = recall_at_k(indices, query_labels, gallery_labels, k=1)
r5 = recall_at_k(indices, query_labels, gallery_labels, k=5)
r10 = recall_at_k(indices, query_labels, gallery_labels, k=10)

print("Recall@1:", r1)
print("Recall@5:", r5)
print("Recall@10:", r10)

import matplotlib.pyplot as plt

metrics = ["Recall@1", "Recall@5", "Recall@10"]
values = [0.10, 0.32, 0.44]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("MASAR Model Performance using Recall@K")
plt.xlabel("Evaluation Metric")
plt.ylabel("Recall Score")

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    return img.permute(1, 2, 0).numpy()

#saliency map
def generate_saliency_map(model, query_img, gallery_img, device):
    model.eval()

    query_img = query_img.unsqueeze(0).to(device)
    gallery_img = gallery_img.unsqueeze(0).to(device)

    # Allow gradients on query image
    query_img.requires_grad_()

    # Extract features
    query_feature = model.query_net(query_img)
    gallery_feature = model.reference_net(x=gallery_img)

    # Normalize features
    query_feature = F.normalize(query_feature, dim=1)
    gallery_feature = F.normalize(gallery_feature, dim=1)

    # Similarity score between drone and satellite
    score = torch.matmul(query_feature, gallery_feature.T)

    # Backpropagate
    model.zero_grad()
    score.backward()

    # Get gradients from query image
    gradients = query_img.grad.data.abs()

    # Convert gradients to heatmap
    saliency, _ = torch.max(gradients, dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    # Normalize heatmap
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency

#XAI
# Choose first query image
query_index = 0

# Get top-1 matched satellite index from FAISS
top1_gallery_index = indices_faiss[query_index][0]

# Get images
query_img, query_label, query_path = query_dataset[query_index]
gallery_img, gallery_label, gallery_path = gallery_dataset[top1_gallery_index]

# Generate saliency
saliency = generate_saliency_map(model, query_img, gallery_img, device)

# Convert image for display
query_display = unnormalize(query_img)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(query_display)
plt.title(f"Drone Query Image\nTrue ID: {query_label}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(query_display)
plt.imshow(saliency, alpha=0.5, cmap="jet")
plt.title(f"XAI Saliency Map\nPredicted ID: {gallery_label}")
plt.axis("off")

plt.tight_layout()
plt.show()
