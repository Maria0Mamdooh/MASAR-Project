from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# Define the batch size for training
batch_size = 8

# Create a DataLoader for the training dataset
# It shuffles the data and uses multiple workers for efficient loading
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

# Initialize the Adam optimizer with a learning rate for model parameters
optimizer = optim.Adam(model.parameters(), lr=1e-5)
# Define the CrossEntropyLoss as the criterion for calculating loss
criterion = nn.CrossEntropyLoss()

# Set the number of training epochs
epochs = 10
# Set the temperature parameter for contrastive learning
temperature = 0.07

# Set the model to training mode
model.train()

# Main training loop
for epoch in range(epochs):
    total_loss = 0

    # Iterate over batches from the training data loader
    for drone_imgs, sat_imgs, ids in train_loader:
        # Move drone and satellite images to the specified device (e.g., GPU)
        drone_imgs = drone_imgs.to(device)
        sat_imgs = sat_imgs.to(device)

        # Pass images through the model to get drone and satellite features
        drone_feat, sat_feat = model(drone_imgs, sat_imgs)

        # Normalize the features to unit length (important for contrastive learning)
        drone_feat = F.normalize(drone_feat, dim=1)
        sat_feat = F.normalize(sat_feat, dim=1)

        # Calculate logits (similarity scores) by multiplying features and dividing by temperature
        # drone_feat and sat_feat.T (transpose) are used to get a similarity matrix
        logits = torch.matmul(drone_feat, sat_feat.T) / temperature
        # Create labels for contrastive loss: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0)).to(device)

        # Calculate the contrastive loss from drone to satellite
        loss_1 = criterion(logits, labels)
        # Calculate the contrastive loss from satellite to drone (transposed logits)
        loss_2 = criterion(logits.T, labels)
        # Combine the two loss components
        loss = (loss_1 + loss_2) / 2

        # Zero the gradients before backpropagation
        optimizer.zero_grad()
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update model parameters using the optimizer
        optimizer.step()

        # Accumulate the loss for the current epoch
        total_loss += loss.item()

    # Calculate the average loss for the epoch
    avg_loss = total_loss / len(train_loader)
    # Print the epoch number and average loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


    # Define the path to save the trained model
save_path = "/content/MASAR_TransGeo_subset.pth"

# Save the model's state dictionary and other training information
torch.save({
    "model_state_dict": model.state_dict(), # The learned weights of the model
    "epoch": epochs,                      # Total epochs trained
    "loss": avg_loss,                     # Final average loss
    "model_name": "MASAR_TransGeo_subset",# Name of the model
    "dataset": "University-1652 subset" # Dataset used for training
}, save_path)

# Print confirmation that the model has been saved
print("Model saved to:", save_path)
