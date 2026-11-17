"""DCGAN generator and discriminator for 64x64 RGB images."""
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, width=64):
        super().__init__()
        self.latent_dim, self.width = latent_dim, width
        blocks = [nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(width * 8), nn.ReLU(True)]
        for source, target in ((8, 4), (4, 2), (2, 1)):
            blocks += [nn.ConvTranspose2d(width * source, width * target, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(width * target), nn.ReLU(True)]
        blocks += [nn.ConvTranspose2d(width, 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.layers = nn.Sequential(*blocks)

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        blocks = [nn.Conv2d(3, width, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)]
        for source, target in ((1, 2), (2, 4), (4, 8)):
            blocks += [nn.Conv2d(width * source, width * target, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(width * target), nn.LeakyReLU(0.2, True)]
        blocks += [nn.Conv2d(width * 8, 1, 4, 1, 0, bias=False)]
        self.layers = nn.Sequential(*blocks)

    def forward(self, image):
        return self.layers(image).flatten()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train_step(generator, discriminator, optim_g, optim_d, real):
    """One discriminator update and one non-saturating generator update."""
    discriminator.requires_grad_(True)
    optim_d.zero_grad(set_to_none=True)
    z = torch.randn(len(real), generator.latent_dim, 1, 1, device=real.device)
    fake = generator(z)
    real_logits, fake_logits = discriminator(real), discriminator(fake.detach())
    criterion = nn.functional.binary_cross_entropy_with_logits
    loss_d = criterion(real_logits, torch.ones_like(real_logits)) + criterion(
        fake_logits, torch.zeros_like(fake_logits))
    if not torch.isfinite(loss_d):
        raise FloatingPointError("Non-finite discriminator loss")
    loss_d.backward()
    optim_d.step()

    discriminator.requires_grad_(False)
    optim_g.zero_grad(set_to_none=True)
    updated_logits = discriminator(fake)
    loss_g = criterion(updated_logits, torch.ones_like(updated_logits))
    if not torch.isfinite(loss_g):
        raise FloatingPointError("Non-finite generator loss")
    loss_g.backward()
    optim_g.step()
    discriminator.requires_grad_(True)
    return dict(loss_g=loss_g.item(), loss_d=loss_d.item(),
                d_real=real_logits.detach().sigmoid().mean().item(),
                d_fake=fake_logits.detach().sigmoid().mean().item())
