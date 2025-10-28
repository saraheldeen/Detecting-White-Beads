# Detecting White Beads

**Author:** Sarah Eldeen  
**Environment:** Python (PyTorch)  
**Goal:** Train a convolutional neural network (CNN) to detect bright, in-focus microbeads used in active microrheology experiments within hydrogels.

---

## 🧠 Overview

This project trains deep learning models to distinguish *good* (bright, in-focus) microbeads from *bad* (out-of-focus) ones in confocal microscopy images. These “good” beads are crucial for accurate stiffness tracking in 3D hydrogels.

The workflow includes:
- Preprocessing and augmenting microscopy image patches  
- Training a custom CNN and a fine-tuned **AlexNet**  
- Predicting bead quality on single images  
- Scanning large images by patches to localize good beads  

---

## 🗂️ Repository Structure

```
Detecting_white_beads/
│
├── Example_Annotated_Dataset/
│   ├── E1/
│   └── E2/
│
├── model_cnn.py           # Defines custom CNN architecture
├── train_cnn.py           # Training loop for CNN
├── train_alexnet.py       # Fine-tuning AlexNet
├── predict_single.py      # Predict bead quality on a single image
├── patch_infer.py         # Patch-level scanning for large images
├── utils.py               # Data loading, transforms, and helper functions
│
├── requirements.txt       # Python dependencies
└── environment.yml        # Conda environment configuration
```

---

## ⚙️ Installation

### Using Conda
```bash
conda env create -f environment.yml
conda activate pytorchenv
```

### Or using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧩 Dataset

Dataset structure:

```
Example_Annotated_Dataset/
│
├── train/
│   ├── good/
│   └── bad/
│
└── test/
    ├── good/
    └── bad/
```

Each image patch should be 300×300 pixels.

---

## 🏋️‍♀️ Training

### Train a Custom CNN
```bash
python train_cnn.py --root Example_Annotated_Dataset --epochs 4 --batch_size 10 --out CNNModel.pt
```

### Fine-Tune AlexNet
```bash
python train_alexnet.py --root Example_Annotated_Dataset --epochs 1 --batch_size 10 --out AlexNetModel.pt
```

---

## 🔍 Predicting

### Predict a Single Image
```bash
python predict_single.py --image path/to/image.jpg --model_path CNNModel.pt
```

Example output:
```
CNNmodel     Predicted value: 0 bad
AlexNetModel Predicted value: 1 good
```

---

## 🧠 Patch Inference on Large Images

To localize good beads in a full 2048×2048 image:
```bash
python patch_infer.py   --image path/to/testimage_2000nm.jpeg   --patch_h 60 --patch_w 60 --step 20   --model_path CNNModel.pt
```

This will:
1. Split the image into 60×60 patches with 20-pixel stride  
2. Predict each patch as *good* or *bad*  
3. Overlay red rectangles on detected good beads  

---

## 📊 Example Results

| Step | Example |
|------|----------|
| **Training Examples** | ![Training Examples](Example_Annotated_Dataset/E1/example_batch.png) |
| **Single Prediction** | ![Single Prediction](Example_Annotated_Dataset/E2/example_prediction.png) |
| **Patch Inference Overlay** | ![Patch Inference](example_results.png) |

---

## 🧩 Model Architecture

Custom CNN (simplified):

```python
self.conv1 = nn.Conv2d(3, 6, 3, 1)
self.conv2 = nn.Conv2d(6, 16, 3, 1)
self.fc1 = nn.Linear(16*73*73, 120)
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 2)
```

AlexNet fine-tuning freezes all layers except the final classifier.

---

## 💡 Notes

- Preprocessing includes random rotation, horizontal flips, resizing, cropping, and normalization using ImageNet means/stds.  
- The environment matches **PyTorch 1.1.0**, **torchvision 0.2.2**, and **Python 3.7.3**.  
- Code is modular and can be easily extended to other microscopy classification tasks.  
