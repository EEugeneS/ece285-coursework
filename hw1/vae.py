"""Convolutional VAE implemented for ECE285 Homework 1."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, image_size=128, width=32):
        super().__init__()
        if image_size not in (32, 64, 128) or width < 8 or width % 8:
            raise ValueError("image_size must be 32/64/128; width a positive multiple of 8")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.config = dict(latent_dim=latent_dim, image_size=image_size, width=width)
        channels = [width * 2**i for i in range(image_size.bit_length() - 3)]
        layers, previous = [], 1
        for channel in channels:
            layers += [nn.Conv2d(previous, channel, 4, 2, 1),
                       nn.GroupNorm(8, channel), nn.SiLU()]
            previous = channel
        self.encoder = nn.Sequential(*layers)
        self.feature_shape = (channels[-1], 4, 4)
        features = channels[-1] * 16
        self.mu = nn.Linear(features, latent_dim)
        self.logvar = nn.Linear(features, latent_dim)
        self.project = nn.Linear(latent_dim, features)
        layers = []
        for channel in reversed(channels[:-1]):
            layers += [nn.ConvTranspose2d(previous, channel, 4, 2, 1),
                       nn.GroupNorm(8, channel), nn.SiLU()]
            previous = channel
        layers += [nn.ConvTranspose2d(previous, 1, 4, 2, 1)]
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        """Return logits; apply sigmoid only for visualization/evaluation."""
        return self.decoder(self.project(z).reshape(-1, *self.feature_shape))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if sample else mu
        return self.decode(z), mu, logvar


def loss_terms(logits, target, mu, logvar, beta=1.0):
    """Sum pixels/latent coordinates per image, then average over the batch."""
    reconstruction = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none").flatten(1).sum(1).mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()
    return reconstruction + beta * kl, reconstruction, kl
