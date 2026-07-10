# Callbacks

A personal grab-bag of custom training callbacks and metrics for Keras/TensorFlow and PyTorch, written for image-segmentation model training (Dice-coefficient-based monitoring, TensorBoard image logging). It is a set of standalone modules to import directly, not an installable package, and is not actively maintained.

## Contents

- `TF2_Callbacks.py` — TensorFlow 2 Keras callbacks and metrics, notably `Add_Images_and_LR`, which writes validation images and the current learning rate to TensorBoard during training.
- `BMA_Callbacks.py` — older TF1-era Keras callbacks: `ModelCheckpoint_new` (checkpointing with a save-best-and-all option and support for multi-GPU-wrapped models), a slightly adapted CyclicLR one-cycle learning-rate callback, and a 3D Dice coefficient helper.
- `PyTorch_Callbacks.py` — `MeanDSC`, a PyTorch `nn.Module` computing the mean Dice similarity coefficient across classes (background class excluded).
- `Visualizing_Model_Utils.py` — `TensorBoardImage`, a subclass of the Keras TensorBoard callback for image logging.

## Requirements

From `requirements.txt`: `numpy`, `tensorflow`, `torchviz`, `PlotScrollNumpyArrays`. The PyTorch module additionally requires `torch`.

## Usage

Import the class you need from the relevant module, e.g. pass `Add_Images_and_LR(log_dir=...)` in the `callbacks` list of `model.fit`, or use `MeanDSC(num_classes=...)` as a metric in a PyTorch training loop.
