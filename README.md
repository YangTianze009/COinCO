# COinCO: Common Inpainted Objects In-N-Out of Context

**Authors:** Tianze Yang\*, Tyson Jordan\*, Ninghao Liu, Jin Sun  
\*Equal contribution  
**Affiliation:** University of Georgia

---

## 🌐 Project Overview

This repository supports the paper:

**"Common Inpainted Objects In-N-Out of Context"**  
_Submitted to NeurIPS 2025 Datasets and Benchmarks Track_  
Status: Submission under review

COinCO is a large-scale dataset derived from COCO, featuring 97,722 images with inpainted objects labeled as **in-context** or **out-of-context** using multimodal reasoning. This dataset enables three novel downstream tasks:

1. **In- and out-of-context classification**
2. **Objects-from-Context Prediction**
3. **Context-empowered fake localization**

---
> 📦 **Dataset Download**: You can download the COinCO dataset from [Hugging Face](https://huggingface.co/datasets/ytz009/COinCO)

##  📁 Repository Structure
```text
├── checkpoints/                         # Pretrained model checkpoints
├── context_prediction/                 # Code for context classification 
├── fake_localization/                  # Code for fake localization
├── objects_from_context_prediction/    # Code for object suggestion task
├── task_data/                          # Data used in these tasks
├── requirements.txt                    # Pip environment file
├── environment.yaml                    # Conda environment file
└── README.md                           # This file
```

---

## ⚙️ Installation Instructions

We recommend setting up a conda environment for full compatibility.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/COinCO.git
cd COinCO
```
### 2. Create and activate a conda environment
```bash
conda env create -f environment.yaml
conda activate COinCO
```
Alternatively, you can use pip:
```bash
pip install -r requirements.txt
```
---
## 📦 Data Usage

The `task_data/` directory contains all necessary **preprocessed data** for the three downstream tasks in our project.

### 🔗 Download Instructions

You can download the required files from our Hugging Face repository:

👉 **[Download link](https://huggingface.co/datasets/ytz009/COinCO-resources)**

Once downloaded, follow these steps to reconstruct and use the dataset:

```bash
# Step 1: Concatenate all parts into a single zip file
cat task_data_part_* > task_data.zip

# Step 2: Unzip the archive
unzip task_data.zip
```

This will create a folder named `task_data/`.

### 📁 Contents of `task_data/`

The `task_data/` folder includes:

- All preprocessed data required to run the code
- Baseline prediction results for the fake localization task (used for context enhancement)

These data files are derived from both the official [COinCO dataset](https://huggingface.co/datasets/ytz009/COinCO) and the original COCO dataset, and have been fully prepared to support out-of-the-box execution of our code.

---
## 🧩 Pretrained Checkpoints

The following contains all pretrained models required for running our experiments.

---

### 🎨 Foundational Models

These external models are required for embedding extraction and context reasoning:

- **[Stable Diffusion 2.1 VAE](https://huggingface.co/stabilityai/stable-diffusion-2-1)** – for extracting image and mask latents.
- **[BERT Base Uncased](https://huggingface.co/bert-base-uncased)** – for encoding semantic information.
- **[Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)** – for multimodal reasoning during context prediction.

---

### 🧠 Our Checkpoints

We provide pretrained models for the two downstream tasks:

```text
checkpoints/
├── context_prediction/
├── objects_from_context_prediction/
```

All checkpoints are bundled and available for download as a single archive:  
👉 **[Download link](https://huggingface.co/datasets/ytz009/COinCO-resources)** 

After downloading the `checkpoints.zip` from Hugging Face
```bash
unzip checkpoints.zip
```
and replace the `checkpoints/` folder

---

## 🚀 Downstream Tasks
### 🔍 1. In- and out-of-context classification
This folder contains the code for training and evaluating models that classify whether an object in a scene is **in-context** or **out-of-context**.

####  1.1 Getting Started

First, navigate to the context prediction folder:

```bash
cd context_prediction
```
####  1.2 Files

- `train.py`: Train the context classification model
- `test.py`: Evaluate the model on the  test set
- `model.py`: Model structure for semantic, visual, and combined input types
- `dataset.py`: Dataset loader for different data sources
- `logs/`: Stores training logs
- `test_results/`: Stores model prediction results on test set

####  1.3 Training
To train a model, run:

```bash
python train.py --device {device_id} --data_source {data_source} --model {model_type}
```

####  Arguments

| Argument       | Type   | Default     | Description                                                                 |
|----------------|--------|-------------|-----------------------------------------------------------------------------|
| `--device`     | `int`  | `0`         | GPU device ID to use (e.g., 0, 1, 2, ...)                                  |
| `--data_source`| `str`  | `balanced`  | Choose the data source:                                                    |
|                |        |             | • `balanced`: Equal number of in-context and out-of-context samples (supplemented with COCO images) |
|                |        |             | • `inpainting_only`: Use only inpainted images from COinCO                 |
| `--model`      | `str`  | `semantic`  | Choose the model type:                                                     |
|                |        |             | • `semantic`: Uses semantic (textual) features only                        |
|                |        |             | • `visual`: Uses visual features only                                      |
|                |        |             | • `combine`: Uses both visual and semantic features                        |

---

####  1.4 Inference / Testing

To evaluate a trained model on the test set:

```bash
python test.py --device {device_id} --data_source {data_source} --model {model_type}
```
#### Arguments
Same as training:

- `--device`: GPU device ID
- `--data_source`: Either `balanced` or `inpainting_only`
- `--model`: `semantic`, `visual`, or `combine`

> 📝 **Note:**  
> The test set is already balanced (1:1 ratio of in-context and out-of-context samples).  
> Evaluation is consistent across all model types and data sources.

---
### 🔍 2. Objects-from-Context prediction
This folder contains the code for training and evaluating models that predict **what objects naturally belong in a given scene**, based on visual context.

---
#### 2.1 Getting Started

Navigate into the folder:

```bash
cd objects_from_context_prediction
```

---

#### 2.2 Files
- `train.py`: Train the object prediction model
- `test.py`: Evaluate the saved checkpoints on the test set
- `model.py`: Model definitions for instance- and clique-level predictions
- `dataset.py`: Dataset loader with context and mask input
- `results/`: Stores evaluation results for all checkpoints

---
#### 2.3 Training

Run the following command to train the model:

```bash
python train.py --device {device_id}
```

####  Argument

| Argument     | Type  | Default | Description                                      |
|--------------|-------|---------|--------------------------------------------------|
| `--device`   | `int` | `0`     | GPU device ID to use (e.g., 0, 1, 2, ...)       |

> ✅ During training, the model will evaluate on the validation set and **automatically save the checkpoints with the highest accuracy, recall, precision, and F1 score**.

---
#### 2.4  Inference / Testing

To evaluate the saved checkpoints on the test set:

```bash
python test.py
```

- All saved checkpoints will be evaluated sequentially.
- Evaluation results will be stored in the `results/` folder.

---

### 🔍 3. Context-empowered Fake Localization

This folder contains the code for enhancing fake localization results using **contextual cues**, specifically by identifying **out-of-context objects** and improving model predictions.

---

#### 3.1 Getting Started

Navigate into the folder:

```bash
cd fake_localization
```

---

#### 3.2 Files

- `context_object_prediction.py`: Uses the Molmo model to find out-of-context objects in the test set.  
  ➤ Results are saved in: `task_data/fake_localization/context_objects_prediction/`

- `evaluation_oracle.py`: Evaluates enhancement performance assuming **ground truth fake objects** are known.  
  ➤ If a fake object is out-of-context (as labeled), enhancement is applied directly to its region.  
  This represents an **upper bound** scenario and typically yields better results than the original predictions.

- `evaluation_ours.py`: Evaluates enhancement performance using **predicted** out-of-context objects (via Molmo).

- `evaluation_results/`: Stores all evaluation outputs.

---

#### 3.3 Running Out-of-Context Object Prediction

To detect out-of-context objects using Molmo:

```bash
python context_object_prediction.py
```

Results will be stored in:

```
task_data/fake_localization/context_objects_prediction/
```

---

#### 3.4 Oracle Evaluation

To evaluate the enhancement, assuming ground truth fake objects are known:

```bash
python evaluation_oracle.py --gamma {gamma_value}
```

- `--gamma`: Context enhancement factor (default: `5`)
- Enhancement is applied only inside known fake object regions.

---

#### 3.5 Ours (Predicted Context) Evaluation

To evaluate the enhancement using **predicted** out-of-context objects:

```bash
python evaluation_ours.py --gamma {gamma_value}
```

- `--gamma`: Context enhancement factor (default: `5`)
- Out-of-context objects are first predicted, and the enhancement is applied accordingly.

---

#### 3.6 Input for Custom Fake Localization Models

The `task_data` directory already includes predictions from four pretrained models:

- [CAT-Net](https://github.com/mjkwon2021/CAT-Net)
- [ManTraNet](https://github.com/ISICV/ManTraNet)
- [Trufor](https://github.com/grip-unina/TruFor)
- [PSCC-Net](https://github.com/proteus1991/PSCC-Net)

To evaluate a different fake localization model:

1. Save your model's predicted fake map as `.npy` files.
2. Place them in:

```
task_data/fake_localization/baselines/
```

3. Run either evaluation script to apply context-based enhancement.

---