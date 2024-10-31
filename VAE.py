import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_data = datasets.FashionMNIST(
    "./data", train=True, download=True, transform=transform
)
train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True)


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.rand_like(std)

    return mean + eps * std


def loss_function(recon_x, x, mean, logvar):
    recon_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return recon_loss + kl_loss


class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = torch.nn.Linear(784, 512)
        self.layer_mean = torch.nn.Linear(512, 2)
        self.layer_logvar = torch.nn.Linear(512, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, z):
        z = self.relu(self.layer1(z))
        mean = self.layer_mean(z)
        logvar = self.layer_logvar(z)

        return mean, logvar


class Decoder(torch.nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = torch.nn.Linear(2, 512)
        self.layer2 = torch.nn.Linear(512, 784)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.layer1(z))
        z = self.sigmoid(self.layer2(z))

        return z


class VAE(torch.nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, z):
        mean, logvar = self.encoder(z)
        z = reparameterize(mean, logvar)

        return self.decoder(z), mean, logvar


def plot_loss_curve(epochs, losses):
    plt.figure()
    plt.plot(range(1, epochs + 1), losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()


def plot_latent_space(model, n_samples=5000):
    model.eval()
    z_points = []
    labels = []
    total = 0

    with torch.no_grad():
        for data, target in train_data_loader:
            data = data.view(-1, 784)
            mean, logvar = model.encoder(data)
            z = reparameterize(mean, logvar)
            z_points.append(z.cpu().numpy())
            labels.append(target.cpu().numpy())

            total += len(labels)
            if total >= n_samples:
                break

    z_points = np.concatenate(z_points)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap="tab10")
    plt.colorbar(scatter)
    plt.title("Latent Space")
    plt.show()


def plot_generated_image(decoder, n_samples=15, grid_range=2):
    grid_x = np.linspace(-grid_range, grid_range, n_samples)
    grid_y = np.linspace(-grid_range, grid_range, n_samples)
    figure = np.zeros((28 * n_samples, 28 * n_samples))

    decoder.eval()

    with torch.no_grad():
        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                z_sample = torch.tensor([[x, y]], dtype=torch.float32)
                figure[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = (
                    decoder(z_sample).cpu().numpy().reshape(28, 28)
                )

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="gray")
    plt.title("Generated Images")
    plt.axis("off")
    plt.show()


def plot_interpolated_image(
    encoder, decoder, data_loader, n_samples=5, n_categories=10
):
    decoder.eval()
    encoder.eval()

    category_images = {i: [] for i in range(n_categories)}
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(-1, 784)
            mean, logvar = encoder(data)
            z = reparameterize(mean, logvar)

            for i in range(len(target)):
                category_images[target[i].item()].append(z[i])
                if all(len(images) >= n_samples for images in category_images.values()):
                    break

            if all(len(images) >= n_samples for images in category_images.values()):
                break

    figure = np.zeros((28 * n_samples, 28 * n_categories))
    with torch.no_grad():
        for i in range(n_categories):
            z_samples = torch.stack(category_images[i])[:n_samples]

            for j in range(n_samples):
                figure[j * 28 : (j + 1) * 28, i * 28 : (i + 1) * 28] = (
                    decoder(z_samples[j].unsqueeze(0))
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(28, 28)
                )

    plt.figure(figsize=(n_categories, n_samples))
    plt.imshow(figure, cmap="gray")
    plt.title("Interpolated Images")
    plt.axis("off")
    plt.show()


model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for data, _ in train_data_loader:
        data = data.view(-1, 784)
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data)

        loss = loss_function(recon_batch, data, mean, logvar)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

    avg = train_loss / len(train_data_loader.dataset)
    losses.append(avg)

    print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_data_loader.dataset):.4f}")

plot_loss_curve(epochs, losses)
plot_latent_space(model)
plot_generated_image(model.decoder)
plot_interpolated_image(model.encoder, model.decoder, train_data_loader)
