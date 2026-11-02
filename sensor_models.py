import torch
import torch.nn as nn

from model import build_resnet20


class SensorMLP(nn.Module):
    """MLP for vector sensor features.

    Attribute names match the same communicated-backbone/private-head boundary
    used by the image model.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x = self.shared_features(x)
        return self.classifier(x)


def build_sensor_mlp(input_dim, num_classes):
    return SensorMLP(input_dim=input_dim, num_classes=num_classes)


def build_model_for_dataset(dataset_name, num_classes, input_dim=None):
    """Build the model architecture associated with a dataset.

    CIFAR-10, MNIST, and FEMNIST use ResNet-20. UCI HAR and PAMAP2 use the
    vector MLP and therefore require ``input_dim``.
    """
    name = dataset_name.lower()
    if name in {"uci_har", "ucihar", "har", "pamap2", "pamap"}:
        if input_dim is None:
            raise ValueError(f"{dataset_name} requires input_dim for SensorMLP")
        return build_sensor_mlp(input_dim=input_dim, num_classes=num_classes)
    return build_resnet20(num_classes=num_classes)
