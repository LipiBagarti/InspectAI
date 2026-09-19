import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

ROOT = Path(r"c:\Users\lipib\Downloads\hackathon\NEU-DET\inspectai_pipeline")
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
IMG_SIZE = 128 # Smaller size for autoencoder

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), # 128x128
            nn.Sigmoid() # Output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Autoencoder on device: {device}")

    # For anomaly detection, we normally train on GOOD images only.
    # Since we lack GOOD images, we will train on a mix of defective images.
    # In a real scenario, this would learn to reconstruct defects, which is bad for anomaly detection.
    # BUT we will use it as a placeholder to demonstrate the architecture.

    transforms_ae = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), # Scales to [0, 1]
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms_ae)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transforms_ae)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs) # Compare output with input
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        reconstruction_errors = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Compute MSE per image
                mse = torch.mean((outputs - inputs)**2, dim=[1,2,3])
                val_loss += mse.sum().item()
                reconstruction_errors.extend(mse.cpu().numpy())
                
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "autoencoder_best.pth")
            
            # Save threshold (95th percentile of validation errors)
            threshold = np.percentile(reconstruction_errors, 95)
            with open(MODELS_DIR / "ae_threshold.txt", "w") as f:
                f.write(str(threshold))
                
            print(f"  --> Saved best model. Threshold set to {threshold:.6f}")

    print("Autoencoder training complete.")

if __name__ == "__main__":
    train_autoencoder()
