import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import os
from jepa_core import SIGRegLoss, VectorFlow

def train():
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LATENT_DIM = 64
    LOG_INTERVAL = 50

    # 1. Define Encoder (Simple CNN for CIFAR-10)
    class Encoder(nn.Module):
        def __init__(self, latent_dim=64):
            super(Encoder, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # 32 -> 16
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2), # 16 -> 8
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), # 8x8 -> 1x1
                nn.Flatten(),
                nn.Linear(64, latent_dim)
            )

        def forward(self, x):
            return self.net(x)

    # 2. Define JEPA Model
    class JEPA(nn.Module):
        def __init__(self, latent_dim):
            super(JEPA, self).__init__()
            self.encoder = Encoder(latent_dim)
            self.predictor = VectorFlow(latent_dim) # From jepa_core.py

        def forward(self, x1, x2):
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            # Predict z2 from z1
            z2_pred = self.predictor(z1)
            return z2_pred, z2

    # 3. Data Loading
    class TwoCropTransform:
        def __init__(self, base_transform):
            self.base_transform = base_transform

        def __call__(self, x):
            return [self.base_transform(x), self.base_transform(x)]

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=TwoCropTransform(transform))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    # 4. Initialization
    model = JEPA(LATENT_DIM).to(DEVICE)
    sig_reg = SIGRegLoss(LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    metrics = []
    global_step = 0

    print(f"Training on {DEVICE}...")

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for i, (images, _) in enumerate(trainloader):
            x1, x2 = images[0].to(DEVICE), images[1].to(DEVICE)
            
            optimizer.zero_grad()
            
            z2_pred, z2 = model(x1, x2)
            
            agreement_loss = torch.mean((z2_pred - z2)**2)
            reg_loss = sig_reg(z2)
            
            loss = agreement_loss + 0.1 * reg_loss
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # 0.0 Acc for unsupervised task
            acc = 0.0

            if global_step % LOG_INTERVAL == 0:
                print(f"Epoch: {epoch+1}, Step: {global_step}, Loss: {loss.item():.4f}, "
                      f"Agreement: {agreement_loss.item():.4f}, SIGReg: {reg_loss.item():.4f}, Acc: {acc}")
                
                metrics.append({
                    'Epoch': epoch + 1,
                    'Step': global_step,
                    'Loss': loss.item(),
                    'Agreement': agreement_loss.item(),
                    'SIGReg': reg_loss.item(),
                    'Acc': acc
                })

    # 6. Save Artifacts
    df = pd.DataFrame(metrics)
    df.to_csv('metrics.csv', index=False)
    
    # Save Covariance Plot
    model.eval()
    with torch.no_grad():
        try:
            # Use a batch to compute covariance
            images, _ = next(iter(trainloader))
            x = images[0].to(DEVICE)
            z = model.encoder(x)
            cov = torch.cov(z.T).cpu().numpy()
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cov)
            plt.colorbar()
            plt.title("Latent Feature Covariance")
            plt.savefig('covariance.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting covariance: {e}")

if __name__ == '__main__':
    train()