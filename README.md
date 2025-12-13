# 🏗️ Concrete Crack Detection using Convolutional Neural Networks (CNNs)

This repository details a Deep Learning project focused on detecting cracks in concrete surfaces. The automatic classification of images allows for the automated detection and monitoring of damage to critical infrastructure.

The primary objective is to develop a robust binary classification model capable of distinguishing between images of "Cracked" and "Non-cracked" concrete, which is vital for automated structural health inspection.

## 🚀 Repository Content Overview

This repository provides all the code, analysis, and documentation necessary to reproduce and understand the project. The core contents cover the entire machine learning workflow:

| Area                      | Key Activities Covered                                                                                                                                                    |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data Analysis**         | Description of the dataset, detailed analysis of the class distribution, visualization, and discussion of any class imbalances.                                           |
| **Data Preparation**      | Creation of Training, Validation, and Test splits. Implementation of a stratified split to maintain proportional class representation across all subsets.                   |
| **Metric Selection**      | Justification for the choice of evaluation metrics (e.g., Accuracy, Sensitivity/Recall, Specificity, F1-Score) based on the project's real-world goals.                    |
| **Augmentation & Loss**   | Formulation of a sensible Data Augmentation Strategy (e.g., relevant transformations for concrete images) and selection of an appropriate Loss Function.                  |
| **Model Development**     | Selection and implementation of an existing, powerful Baseline Network (e.g., a variant of ResNet) and the Design and Justification of a Custom CNN Architecture.          |
| **Performance Analysis**  | Quantitative comparison of the baseline and custom models. Analysis of training curves and interpretation of results.                                                   |
| **Experimentation**       | Studies on the influence of various parameters (e.g., Learning Rate, Batch Size, Data Augmentation intensity, network size) on the final performance metrics.            |
| **Error Analysis**        | Examination of specific cases where the network failed (misclassifications). Discussion and hypothesis on the potential causes of these errors.                           |
| **Explainability (Optional)** | Visual analysis (e.g., using Grad-CAM) to determine which specific image regions the network attends to when making a crack detection decision.                         |

## 🛠️ Setup & Environment

This project is designed to be run on **Google Colab**, which provides free access to the necessary computational resources, including GPUs.

-   **Efficient Training**: Deep Learning models can be trained efficiently using GPUs. Google Colab offers free GPU access (like the T4 GPU), which is essential for this project.
-   **Notebooks**: The entire project is contained within a Python notebook (`.ipynb`), which can be easily created and executed on Google Colab.
-   **Prerequisite**: You will need a Google Account to use Google Colab.

To get started, simply open the notebook in Colab and ensure you have a GPU runtime selected:
1.  Navigate to **Runtime** -> **Change runtime type**.
2.  Select **T4 GPU** (or another available GPU) from the "Hardware accelerator" dropdown menu.
3.  You can access the platform here: [https://colab.research.google.com](https://colab.research.google.com)

## � Requirements

The following Python packages are required to run the notebooks in this project. The versions listed are those used in the development environment:

- numpy==2.3.5
- pandas==2.3.3
- matplotlib==3.10.8
- pillow==12.0.0
- torch==2.7.1+cu118
- torchvision==0.22.1+cu118
- scikit-learn==1.8.0
- seaborn==0.13.2

You can install these packages using pip:

```bash
pip install numpy==2.3.5 pandas==2.3.3 matplotlib==3.10.8 pillow==12.0.0 torch==2.7.1+cu118 torchvision==0.22.1+cu118 scikit-learn==1.8.0 seaborn==0.13.2
```

Note: PyTorch versions with CUDA support (cu118) are specified for GPU acceleration. If running on CPU-only systems, you may need to adjust the PyTorch installation accordingly.

## �📝 Evaluation Criteria

The project will be evaluated based on the following criteria:

1.  **Dataset Analysis (15 Points)**
    -   Description, class distribution, visualization, discussion of imbalances.
2.  **Data Preparation & Splitting (10 Points)**
    -   Stratified split, justification for the split, consideration of subgroups.
3.  **Choice & Justification of Metrics (10 Points)**
    -   Meaningful metrics, justification regarding the goal (e.g., sensitivity, specificity).
4.  **Data Augmentation Strategy (10 Points)**
    -   Selection of suitable methods, justification, and implementation in code.
5.  **Choice of Loss Function (5 Points)**
    -   Appropriate loss function with a brief justification.
6.  **Model Selection / Baseline Network (10 Points)**
    -   Selection and justification of an existing model (e.g., ResNet).
7.  **Custom Model Design (15 Points)**
    -   Architecture & justification of a custom network.
8.  **Performance Analysis & Comparison (10 Points)**
    -   Quantitative comparison, training curves, interpretation.
9.  **Parameter Studies & Experiments (10 Points)**
    -   Variation of parameters (e.g., Learning Rate, batch size), analysis of effects.
10. **Error Analysis (5 Points)**
    -   Examples of misclassifications, discussion of possible causes.
11. **Bonus: Explainability / Visualization (+5 Points)**
    -   Optional: e.g., using Grad-CAM.

