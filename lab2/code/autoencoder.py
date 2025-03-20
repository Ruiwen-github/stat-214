import lightning as L
import torch
import torch.nn as nn


class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        # Below is the definition of the encoder and decoder.

        input_size = int(n_input_channels * (patch_size**2))

        # Use deeper model

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),  # Normalization
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embedding_size)  # Bottleneck
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, input_size),
            torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size))
        )

    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # All the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # Encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # The loss is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)

        # Log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # Encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # The loss is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(batch, decoded)
        # Log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # Set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)

class ConvAutoencoder(L.LightningModule):

    """
    Convolutional Autoencoder implemented using PyTorch Lightning.
    This autoencoder compresses the input data into a lower-dimensional 
    embedding and reconstructs it back to its original form.
    """
    def __init__(self, optimizer_config=None, patch_size=9, n_input_channels=8, embedding_size=8):
        super().__init__()
        if optimizer_config is None:
            optimizer_config = {}

        self.optimizer_config = optimizer_config
        self.patch_size = patch_size

        # Encoder: Convolutional layers to reduce spatial dimensions and extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),  # patch_size -> (patch_size - 1) / 2 + 1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Further downsampling
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        # Dynamically calculate flattened size after convolution
        with torch.no_grad():
            dummy_input = torch.randn(1, n_input_channels, self.patch_size, self.patch_size)
            dummy_output = self.encoder(dummy_input)
            self.flattened_size = int(torch.flatten(dummy_output, start_dim=1).shape[1])

        # Fully connected layer to map to latent space
        self.fc1 = nn.Linear(self.flattened_size, embedding_size)

        # Fully connected layer to map back to feature space
        self.fc2 = nn.Linear(embedding_size, self.flattened_size)

        # Decoder: Transposed convolutional layers to reconstruct the input
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, n_input_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid(),   # Output in range [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_input_channels, patch_size, patch_size)

        Returns:
            torch.Tensor: Reconstructed tensor with the same shape as input.
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 256, 3, 3)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.

        Args:
            batch (torch.Tensor): Batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss value.
        """
        encoded = self.encoder(batch)
        decoded = self.forward(batch)
        loss = torch.nn.functional.mse_loss(batch, decoded)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.

        Args:
            batch (torch.Tensor): Batch of input data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss value.
        """
        encoded = self.encoder(batch)
        decoded = self.forward(batch)
        loss = torch.nn.functional.mse_loss(batch, decoded)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Generate an embedding for the input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_input_channels, patch_size, patch_size)

        Returns:
            torch.Tensor: Embedding of shape (batch_size, embedding_size)
        """
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x