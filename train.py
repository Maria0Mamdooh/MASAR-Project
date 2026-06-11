import torch
from model.TransGeo import TransGeo

class Args:
    dim = 512
    dataset = "university"
    sat_res = 0
    fov = 0
    crop = False

args = Args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = TransGeo(args).to(device)
model.eval()

sample_drone = sample_drone.unsqueeze(0).to(device)
sample_sat = sample_sat.unsqueeze(0).to(device)

with torch.no_grad():
    drone_feat, sat_feat = model(sample_drone, sample_sat)

print("Drone feature shape:", drone_feat.shape)
print("Satellite feature shape:", sat_feat.shape)

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

epochs = 20
temperature = 0.07

model.train()

for epoch in range(epochs):
    total_loss = 0

    for drone_imgs, sat_imgs, ids in train_loader:
        drone_imgs = drone_imgs.to(device)
        sat_imgs = sat_imgs.to(device)

        drone_feat, sat_feat = model(drone_imgs, sat_imgs)

        drone_feat = F.normalize(drone_feat, dim=1)
        sat_feat = F.normalize(sat_feat, dim=1)

        logits = torch.matmul(drone_feat, sat_feat.T) / temperature
        labels = torch.arange(logits.size(0)).to(device)

        loss_1 = criterion(logits, labels)
        loss_2 = criterion(logits.T, labels)
        loss = (loss_1 + loss_2) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    scheduler.step()

save_path = "/content/MASAR_TransGeo_subset.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "epoch": epochs,
    "loss": avg_loss,
    "model_name": "MASAR_TransGeo_subset",
    "dataset": "University-1652 subset"
}, save_path)

print("Model saved to:", save_path)
