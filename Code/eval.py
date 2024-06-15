"""This is a script for evaluating your trained model.
 
This is just a starting point for your validation pipeline.
"""
 
import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
 
import mlflow
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2
import numpy as np
import sys
 
ROOT_DIR = Path(__file__).parent
 
# Set the tracking server to be localhost with sqlite as tracking store
mlflow.set_tracking_uri(uri="sqlite:///mlruns.db")
 
 
class SafetyBatchDataset(torchvision.datasets.ImageFolder):
    """Custom dataset for safety batch."""
 
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(SafetyBatchDataset, self).__init__(
            root=root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
 
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        # convert target index to class label
        target_label = int(
            [label for label, idx in self.class_to_idx.items() if idx == target][0]
        )
 
        return sample, target_label
 
 
def get_sign_names() -> Dict[int, str]:
    """Gets the corresponding sign names for the classes."""
    sign_names_file_path = Path(__file__).parent / "signnames.csv"
    sign_names = {}
    with open(sign_names_file_path, mode="r") as sign_names_file:
        sign_names_reader = csv.reader(sign_names_file)
        next(sign_names_reader, None)  # skip the header
        for line in sign_names_reader:
            class_id = int(line[0])
            sign_name = line[1]
            sign_names[class_id] = sign_name
 
    return sign_names
 
 
def load_and_transform_data(
    data_directory_path: str,
    batch_size: int = 64,
    img_dimensions: tuple[int, int] = (32, 32),
) -> torch.utils.data.DataLoader:
    """Loads data from directory, resizes and rescales images to floats
    between 0 and 1.
 
    You may want to extend this.
    """
    data_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(img_dimensions),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomRotation(degrees=(-15,15)),
            v2.ColorJitter(brightness=(0,0.1), hue=(-0.1,0.1)),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1,0.3)),
        ]
    )
 
    dataset = SafetyBatchDataset(data_directory_path, transform=data_transforms)
 
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )
 
    return data_loader
 
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
 
def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model: nn.Module,
                                n_classes,
                                n_samples):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes
 
    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
 
    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):
            #image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output)  # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))
 
        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)
 
    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
 
    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
 
    mean_value = np.mean(mean)
    variance_value = np.mean(variance)
 
    print(
     f"Safety batch: mean epistemic uncertainty: {mean_value:.4f}, variance epistemic uncertainty: {variance_value:.4f}"
    )
 
    return mean_value, variance_value
 
def evaluate(
    model: nn.Module,
    loss_function: nn.modules.loss,
    batch_loader: torch.utils.data.DataLoader,
    batch_name: str,
) -> List[int]:
    """Evaluates the model on the validation batch.
 
    You may want to extend this to report more metrics. However, it is not about
    how many metrics you crank out, it is about whether you find the meaningful
    ones and report. Think thoroughly about which metrics to go for.
    """
    model.eval()
    batch_loss = 0
    correct = 0
    predictions = []
 
    true_positives = 0
    false_positives = 0
 
    for data, target in batch_loader:
        with torch.no_grad():
            output = model(data)
            batch_loss += loss_function(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            predictions.extend(predicted.tolist())
 
             #newcommand definition true_positives and false_positives
            true_positives += ((predicted == target) & (predicted == 1)).sum().item()
            false_positives += ((predicted != target) & (predicted == 1)).sum().item()
 
    batch_loss /= len(batch_loader.dataset)
    batch_accuracy = correct / len(batch_loader.dataset)
 
    #newcommand equation for validation_precision
    batch_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
 
    print(
        f"Safety batch: Average loss: {batch_loss:.4f}, Accuracy: {100.0 * batch_accuracy:.1f} %, Precision: {100.0 * batch_precision:.1f} %"
    )
   
    #newcommand
    mlflow.log_metric("evaluation accuracy", batch_accuracy)
    mlflow.log_metric("evaluation loss", batch_loss)
   
    #newcommand
    mlflow.log_metric("validation_precision", batch_precision)
 
    return predictions
 
 
if __name__ == "__main__":
    # you may want to use different parameters than the default ones
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "safetyBatches" / "Batch_1"),
        help="Directory path where evaluation batch is located.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="MLFlow Run ID which contains a logged model to evaluate.",
    )
 
    args = parser.parse_args()
 
    # Load the logged model and evaluate it
    model_uri = f"runs:/{args.run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)
 
    criterion = nn.CrossEntropyLoss()
 
    batch_loader = load_and_transform_data(data_directory_path=args.data_dir)
 
    predictions = evaluate(loaded_model, criterion, batch_loader)
 
    # Output incorrect classifications
    ground_truth = []
    for _, target in batch_loader:
        ground_truth.extend(target.tolist())
    sign_names = get_sign_names()
    wrong_predictions_idx = [
        idx
        for idx, (y_pred, y) in enumerate(zip(predictions, ground_truth))
        if y_pred != y
    ]
    for idx in wrong_predictions_idx:
        print(
            f"Traffic sign {sign_names[ground_truth[idx]]} incorrectly "
            f"classified as {sign_names[predictions[idx]]}"
        )
 