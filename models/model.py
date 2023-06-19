from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class MotionDetect(L.LightningModule):
    """
    Model used for leraning from motion data
    """

    def __init__(
        self,
        img_size: int,
        num_classes: int,
        kernel_size: int = 9,
        pool_size: int = 2,
        learning_rate: float = 0.01,
        conv1_out_channels: int = 6,
        conv2_out_channels: int = 10,
    ) -> None:
        """
        Initializes the MotionDetect class
        :param img_size: Size of the input image
        :param num_classes: Number of output categories
        :param kernel_size: size of the kernel in convolutional layers
        """

        super().__init__()

        self._learning_rate = learning_rate
        self._loss_fn = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

        img_size_1 = int((img_size - kernel_size + 1) / pool_size)
        self._final_img_size = int((img_size_1 - kernel_size + 1) / pool_size)
        self._conv2_out_channels = conv2_out_channels

        self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size)
        self.fc1 = nn.Linear(conv2_out_channels * self._final_img_size**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._conv2_out_channels * self._final_img_size**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self._loss_fn(output_i, label_i)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Returns Tuple of predicted class and true label of that class
        """

        input_i, label_i = batch
        prediction = self.forward(input_i)
        loss = self._loss_fn(prediction, label_i)
        self.log("validation_loss", loss)
        predicted_class = torch.argmax(prediction, dim=-1)
        self.validation_step_outputs.append((predicted_class, label_i))

        return predicted_class, label_i

    def on_validation_epoch_end(self) -> None:
        predictions = torch.cat([pred for pred, _ in self.validation_step_outputs])
        targets = torch.cat([target for _, target in self.validation_step_outputs])

        accuracy = torch.sum(predictions == targets).item() / len(
            self.validation_step_outputs
        )

        self.log("validation_accuracy", accuracy)
        self.validation_step_outputs.clear()
