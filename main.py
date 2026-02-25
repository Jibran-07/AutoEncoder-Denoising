import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.decoder(self.encoder(x)).view(-1, 1, 28, 28)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_loader = DataLoader(
    torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True),
    batch_size=128, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True),
    batch_size=128
)

model = DenoisingAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for data, _ in train_loader:
        noisy = data + torch.randn_like(data) * 0.3
        noisy = torch.clamp(noisy, -1, 1)

        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.4f}')

model.eval()
with torch.no_grad():
    test_data, _ = next(iter(test_loader))
    original = test_data[0]
    noisy = original + torch.randn_like(original) * 0.3
    noisy = torch.clamp(noisy, -1, 1)
    denoised = model(noisy.unsqueeze(0))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original.squeeze(), cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(noisy.squeeze(), cmap='gray')
plt.title('Noisy')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(denoised.squeeze(), cmap='gray')
plt.title('Denoised')
plt.axis('off')
plt.show()






