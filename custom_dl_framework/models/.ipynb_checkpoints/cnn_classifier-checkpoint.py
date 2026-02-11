# models/cnn_classifier.py

from models.base_model import BaseModel
from layers.conv2d import Conv2D
from layers.activation_relu import ReLU
from layers.max_pool2d import MaxPool2D
from layers.dense import Dense
from core.tensor import Tensor


class CNNClassifier(BaseModel):
    """
    CNN with true multi-channel Conv2D and two Linear layers:
    576 → 50 → num_classes
    """

    def __init__(self, num_classes):
        # -------- Conv Block 1 --------
        self.conv1 = Conv2D(
            in_channels=3,
            out_channels=10,
            kernel_size=5,
            stride=1
        )
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        # -------- Conv Block 2 --------
        self.conv2 = Conv2D(
            in_channels=10,
            out_channels=20,
            kernel_size=5,
            stride=1
        )
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        # Feature map size:
        # [16][6][6] → 576

        # -------- Fully Connected Layers --------
        self.fc1 = Dense(
            in_features=500,
            out_features=50
        )

        self.fc2 = Dense(
            in_features=50,
            out_features=num_classes
        )

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.fc1, self.fc2
        ]

    def forward(self, x):
        # -------- Conv Block 1 --------
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        # -------- Conv Block 2 --------
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        # -------- Flatten --------
        flat = []
        for c in x.data:
            for row in c:
                for val in row:
                    flat.append(val)

        x_flat = Tensor(flat, requires_grad=True)

        # -------- Fully Connected --------
        x = self.fc1.forward(x_flat)

        # IMPORTANT: Dense outputs list[Tensor] → pack into Tensor
        x_fc1 = Tensor([t.data for t in x], requires_grad=True)

        return self.fc2.forward(x_fc1)
