import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 64
LATENT_DIM = 100
N_CRITIC = 5
N_EPOCHS = 50
LEARNING_RATE = 1e-4
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_data = datasets.FashionMNIST(
    "./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


class Generator(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, z):
        return self.model(z)


def get_gradient_penalty(discriminator, real_images, gen_images, lambda_gp=10):
    epsilon = torch.rand(real_images.size(0), 1, 1, 1)
    x_hat = epsilon * real_images + (1 - epsilon) * gen_images
    x_hat.requires_grad = True

    d_x_hat = discriminator(x_hat)
    gradients = torch.autograd.grad(
        inputs=x_hat,
        outputs=d_x_hat,
        grad_outputs=torch.ones(d_x_hat.size()),
        create_graph=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


generator = Generator(latent_dim=LATENT_DIM)
discriminator = Discriminator()

optim_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optim_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

wasserstein_losses = []
gp_losses = []
total_losses = []
g_losses = []

for epoch in range(N_EPOCHS):
    for i, (real_images, _) in enumerate(train_loader):
        for _ in range(N_CRITIC):
            z = torch.randn(real_images.size(0), LATENT_DIM)
            gen_images = generator(z).detach()

            d_fake = discriminator(gen_images).mean()
            d_real = discriminator(real_images).mean()
            gradient_penalty = get_gradient_penalty(
                discriminator=discriminator,
                real_images=real_images,
                gen_images=gen_images,
            )

            wasserstein_loss = d_fake - d_real
            d_loss = wasserstein_loss + gradient_penalty

            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()

        z = torch.randn(real_images.size(0), LATENT_DIM)
        g_loss = -discriminator(generator(z)).mean()

        optim_g.zero_grad()
        g_loss.backward()
        optim_g.step()

        wasserstein_losses.append(wasserstein_loss.item())
        gp_losses.append(gradient_penalty.item())
        total_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if i % 100 == 0:
            print(
                f"Epoch [{epoch}/{N_EPOCHS}], Step [{i}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
            )


epochs = list(range(len(wasserstein_losses)))
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(epochs, wasserstein_losses, label="Wasserstein Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Wasserstein Loss vs Epochs")

plt.subplot(1, 3, 2)
plt.plot(epochs, gp_losses, label="Gradient Penalty Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Gradient Penalty Loss vs Epochs")

plt.subplot(1, 3, 3)
plt.plot(epochs, total_losses, label="Total Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Total Loss vs Epochs")
plt.show()


def generate_images(generator, latent_dim, num_images=100):
    generator.eval()
    z = torch.randn(num_images, latent_dim)
    with torch.no_grad():
        gen_images = generator(z)

    gen_images = gen_images.view(gen_images.size(0), 28, 28).numpy()
    _, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(gen_images[i * 10 + j], cmap="gray")
            axes[i, j].axis("off")

    plt.show()


generate_images(generator, LATENT_DIM)
