from .tensor import Tensor, Parameter
from .model import G16BuildModel, SimpleCNN
from .loss import CrossEntropyLoss
from .optimizer import SGD
from .dataloader import ImageFolderDataset, DataLoader, SubsetDataset, G16TrainValSplit
from .metrics import G16BatchAccuracy
from .serialization import G16SaveWeights, G16LoadWeights, G16ReadMetadata
from .backend import G16GetBackend

# Backward-compatible aliases
train_val_split = G16TrainValSplit
batch_accuracy = G16BatchAccuracy
save_weights = G16SaveWeights
load_weights = G16LoadWeights
read_metadata = G16ReadMetadata
get_backend = G16GetBackend

__all__ = [
    "Tensor",
    "Parameter",
    "SimpleCNN",
    "G16BuildModel",
    "CrossEntropyLoss",
    "SGD",
    "ImageFolderDataset",
    "SubsetDataset",
    "G16TrainValSplit",
    "train_val_split",
    "DataLoader",
    "G16BatchAccuracy",
    "batch_accuracy",
    "G16SaveWeights",
    "save_weights",
    "G16LoadWeights",
    "load_weights",
    "G16ReadMetadata",
    "read_metadata",
    "G16GetBackend",
    "get_backend",
]
