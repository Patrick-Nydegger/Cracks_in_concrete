# 🔬 Experiment Journal: 
# Use case: Automatic inspection of bridge piers and detection of cracks in concrete

---

## **Project Details**

> **Institution:** 🏛️ FHNW School of Business
>
> **Module:** 📚 Deep Learning (HS 2025)
>
> **Authors:** 👥 Oliver Gwerder, Patrick Nydegger
>
> **Date:** 📅 October - December 2025
>
> **Weight:** ⚖️ 30% of the final module grade

---
## Main Objective

> To develop and evaluate a Convolutional Neural Network (CNN) capable of accurately classifying images of concrete surfaces as either "Cracked" or "Non-cracked".

### Use Case

> A drone flies along a predefined course around an object (e.g., bridge piers). It uses a camera to continuously search for cracks in the concrete. The deep learning model automatically processes these camera images in real time, possibly on the drone itself, and classifies the images as “positive” or “negative.” This means that we need a model that is as fast and small as possible, since the drone has a limited payload and only a short flight time, so speed plays a major role. As soon as an image is classified as “positive,” the drone marks it with a spray can. The images are stored together with the GPS location. After the flight, the images are validated by a potentially more complex model and possibly further classified. The goal is to achieve a complete, seamless recording of all damage.
> For this project work, the focus was on the classification model that performs the initial assessment, possibly directly on the drone.

## Project Summary

> This project involves the entire machine learning workflow, from data analysis and preprocessing to the implementation of a baseline model and a custom-designed CNN. We will document our experiments, compare model performance using appropriate metrics, and analyze the results to determine the most effective approach for automated crack detection.

---

## 📋 Project Checklist & Table of Contents

- [ ] 1. Dataset Description and Analysis
- [ ] 2. Data Splitting Strategy
- [ ] 3. Choice of Evaluation Metrics
- [ ] 4. Data Augmentation Strategy
- [ ] 5. Choice of Loss Function
- [ ] 6. Baseline Model Selection
- [ ] 7. Custom Model Design
- [ ] 8. Performance Analysis
- [ ] 9. Parameter Studies & Experiments
- [ ] 10. Error Analysis (Failure Cases)
- [ ] 11. (Bonus) Explainability Analysis

---

### 1. Dataset Description and Analysis
*   **Dataset Source:**
*   **Content:**
*   **Image Properties:**
    *   Dimensions:
    *   Color Space:
    *   Total number of images:
*   **Class Distribution Analysis:**
    *   **Class "Cracked":**
    *   **Class "Non-Cracked":**
    *   **Imbalance:**
    *   **Visualization:**

### 2. Data Splitting Strategy
*   **Existing Split:**
*   **Splitting Method:**
    *   **Ratio:**
    *   **Stratification:**
*   **Final Split Counts:**
    *   **Training Set:**
    *   **Validation Set:**
    *   **Test Set:**

### 3. Choice of Evaluation Metrics
We base our evaluation of key metrics on the use case. To do this, we perform a brief risk assessment.
Consequences of errors:
False negative (FN – genuine crack overlooked): Critical error. This means that a potentially structurally dangerous crack is not documented and remains untreated. The consequence is a high safety risk.

False positive (FP – no crack is marked as a crack): Non-critical error. This only leads to an unnecessary manual recheck at this point. The consequence is higher operating costs, but no safety risk.

Our primary metric that we want to optimize is therefore recall.
By achieving the highest possible recall, we minimize the risk of a crack not being detected and becoming a safety risk.

We choose precision as our secondary metric to ensure that the workflow is not overloaded with too many unnecessary checkpoints.
Accuracy could potentially be used for training, but this is somewhat risky, as in reality many images probably do not contain cracks. Thus, if the model rated all images as “negative,” there would be a high level of accuracy.

### 4. Data Augmentation Strategy
*   **Necessity:**
*   **Selected Techniques & Justification:**
*   
  - [ ] Horizontal/Vertical Flips
  - [ ] Rotations
  - [ ] Brightness/Contrast Adjustments
  - [ ] Zoom
    *   `[ ]` Rotations
    *   `[ ]` Brightness/Contrast Adjustments
    *   `[ ]` Zoom
 
*   

### 5. Choice of Loss Function
*   **Selected Loss Function:**
*   **Justification:**

### 6. Baseline Model Selection
*   **Chosen Architecture:**
*   **Reason for Choice:**

### 7. Custom Model Design
*   **Architecture Overview:**
    *   Number of convolutional layers:
    *   Activation functions used:
    *   Pooling layers:
    *   Regularization:
    *   Classifier head:
*   **Design Justification:**

### 8. Performance Analysis
*   **Comparison Table:**

    | Model         | Accuracy | F1-Score | Recall | Precision |
    |---------------|----------|----------|--------|-----------|
    | **Baseline**  |          |          |        |           |
    | **Custom CNN**|          |          |        |           |

* **Training Curves:**
*   **Interpretation:**

### 9. Parameter Studies & Experiments
*   **Objective:**
*   **Experiment 1: Learning Rate Tuning**
*   **Experiment 2: Batch Size**
*   **Experiment 3: Data Augmentation Intensity**

### 10. Error Analysis (Failure Cases)
*   **Analysis of Misclassifications:**
    *   **False Positives (Non-Cracked predicted as Cracked):**
    *   **False Negatives (Cracked predicted as Non-Cracked):**
*   **Hypothesis:**

### 11. (Bonus) Explainability Analysis
*   **Method Used:**
*   **Findings:**
*   **Insights:**
