# 🔬 Experiment Journal: 
# Use case: Automatic inspection of bridge piers and detection of cracks in concrete


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

✅ 1. Dataset Description and Analysis

✅ 2. Data Splitting Strategy

✅ 3. Choice of Evaluation Metrics

✅ 4. Data Augmentation Strategy

✅ 5. Choice of Loss Function
- [ ] 6. Baseline Model Selection
- [ ] 7. Custom Model Design
- [ ] 8. Performance Analysis
- [ ] 9. Parameter Studies & Experiments
- [ ] 10. Error Analysis (Failure Cases)
- [ ] 11. (Bonus) Explainability Analysis

---

### 1. Dataset Description and Analysis

The dataset used for this project is the "Concrete Crack Images for Classification" dataset, sourced from Kaggle. 
[https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification/data]

It contains a total of 40,000 images of concrete surfaces, collected from various buildings on the METU Campus. The images are provided in RGB format with a resolution of 227x227 pixels. The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016).
No data augmentation in terms of random rotation or flipping was applied.


#### The data is pre-labeled and divided into two distinct classes:

Positive: Images containing a visible crack.
Negative: Images of concrete surfaces without any cracks.

#### Visual analysis of sample images reveals some key characteristics:

**Image Quality:** The images vary in terms of lighting, surface texture, and coloration.
**Crack Variety:** Cracks in the 'Positive' samples range from very fine, hairline fractures to large, obvious fissures.
**Potential Challenges:** Some 'Negative' samples contain features like shadows, stains, or joints in the concrete that could visually mimic cracks. This ambiguity is expected to be the primary challenge for our classification model.


<img width="1107" height="575" alt="image" src="https://github.com/user-attachments/assets/33d32723-cd46-436f-92cc-4e04301753ad" />


#### Our analysis confirms that the dataset consists of exactly 20,000 images for the 'Positive' class and 20,000 images for the 'Negative' class.
 
The dataset is perfectly balanced, with a 50/50 split between the two classes. This is an ideal scenario for a binary classification task.

#### Implications of this balance:

There is a lower risk of the model developing a bias towards a majority class, which often happens in imbalanced datasets.
Since the dataset is well-balanced, accuracy can reliably be used as a performance metric.
We do not need to employ complex techniques to handle class imbalance, such as oversampling or undersampling.
 
<img width="713" height="547" alt="image" src="https://github.com/user-attachments/assets/54126223-27e3-4544-85d2-3be8ece9761f" />



___


### 2. Data Splitting Strategy
#### Existing Split:
The original dataset from Kaggle does not provide a pre-defined training, validation, or test split. It only provides two folders, Positive and Negative, containing all 40,000 images.
#### Splitting Method:
We have partitioned the dataset into three subsets with the following ratio:

**Training Set: 70% of the data**

   -> The training set is deliberately made the largest partition. This provides the model with a rich and diverse set of 28,000 examples, which is crucial for learning the complex and subtle features that distinguish cracked from non-cracked concrete surfaces.
   
**Validation Set: 15% of the data**

   -> The validation set (6,000 images) serves as a proxy for unseen data during the training phase. It is used to monitor the model's performance epoch by epoch, allowing us to detect overfitting and to make informed decisions about hyperparameter tuning (e.g., adjusting the learning rate or deciding on the number of epochs). 
  
**Test Set: 15% of the data**

   -> The test set (6,000 images) is the final holdout set. It remains completely untouched during the entire development and tuning process. Its sole purpose is to provide a single, final, and unbiased assessment of our best model's performance on completely new data, simulating its deployment in a real-world scenario.


**Final Split:**

```
Training =      Positive: 14000 + Negative: 14000   -> Total Training:     28000

Validation =    Positive: 3000 + Negative: 3000     -> Total Validation:   6000

Test =          Positive: 3000 + Negative: 3000     -> Total Test:         6000
```

### 3. Choice of Evaluation Metrics test
We base our evaluation of key metrics on the use case. To do this, we perform a brief risk assessment.

#### Consequences of errors:

False negative (FN – genuine crack overlooked): Critical error. This means that a potentially structurally dangerous crack is not documented and remains untreated. The consequence is a high safety risk.

False positive (FP – no crack is marked as a crack): Non-critical error. This only leads to an unnecessary manual recheck at this point. The consequence is higher operating costs, but no safety risk.

Our primary metric that we want to optimize is therefore **recall**.
   -> By achieving the highest possible recall, we minimize the risk of a crack not being detected and becoming a safety risk.

We choose **Specificity** as our secondary metric. This measures the model's ability to correctly identify non-cracked surfaces. A high specificity means a low number of False Positives. While less critical than Recall, a reasonably high Specificity is still desirable to keep the cost of unnecessary manual inspections low.

Further we take a look at: 
* Accuracy
* F1-Score

### 4. Data Augmentation Strategy
#### Necessity:
Data augmentation is essential to bridge the gap between our training data and the dynamic reality of drone-based inspection. Since the model will be deployed on a flying drone, the distance to the concrete surface will constantly fluctuate, resulting in varying image resolutions and scales. Furthermore, outdoor weather conditions introduce unpredictable changes in brightness and contrast. Additionally, cracks and defects naturally occur in arbitrary orientations. By simulating these specific variances—such as scaling, brightness adjustments, and random rotations—we ensure the model is robust enough to handle the unstable conditions of a real-world flight.

#### Selected Techniques & Justification:
##### Geometric Transformations (Simulating Physical Variations):

**RandomResizedCrop(size=224, scale=(0.8, 1.0)):** This is a powerful, compound transformation that addresses two key challenges simultaneously. By randomly cropping a region of the image (between 80% and 100% of the original area) and resizing it back to 224x224, it effectively simulates:

- **Zooming:** Simulates variations in the distance between the camera and the concrete surface.
- **Shifting (Translation):** Since the crop is randomly positioned, it ensures that cracks are not always centered, forcing the model to detect them anywhere in the frame.
  
**RandomHorizontalFlip & RandomVerticalFlip (p=0.5):** A crack's classification is independent of its orientation. These flips teach the model this fundamental invariance.

**RandomRotation(degrees=45):** This simulates variations in camera angle, making the model robust to inspections from non-parallel viewpoints.

##### Color Transformations (Simulating Environmental Variations):

**ColorJitter(brightness=0.3, contrast=0.3):** Lighting is highly unpredictable in field inspections. Altering brightness and contrast forces the model to learn the structural shape of a crack rather than relying on specific pixel intensities, making it robust to environmental changes.



<img width="1385" height="955" alt="image" src="https://github.com/user-attachments/assets/4235fed6-927a-4805-b95b-4ac8d9616278" />


### 5. Choice of Loss Function

The choice of loss function is being strategically managed to align the model's training objective with the critical project goal: minimizing dangerous errors.

#### 1. Baseline: Binary Cross-Entropy (BCE)
The initial model will use Binary Cross-Entropy (BCE).
BCE is the standard, mathematically robust choice for binary classification (crack vs. no crack). It will serve to establish a reliable, well-calibrated performance benchmark, ensuring the fundamental model architecture is sound before specialized adjustments are made.

#### 2. The Contextual Challenge: Asymmetric Risk
The dataset's error costs are asymmetric: a False Negative (missing a crack) is catastrophically more expensive than a False Positive (a false alarm).
This necessitates optimizing the model for high Sensitivity (Recall)—its ability to correctly identify all actual crack cases.

#### 3. Advanced Solution for Experiments in Chapter 9: Weighted BCE
To address the asymmetric risk, the strategy includes an experimental pivot to Weighted Binary Cross-Entropy (Weighted BCE).
Weighted BCE uses a pos_weight parameter to assign a significantly higher penalty to errors involving the critical 'Positive' (crack) class.
This directly forces the training process to focus on driving down the rate of dangerous False Negatives, thus directly tailoring the model's optimization to achieve the project's paramount safety objective.


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

    | Model             | Recall    | Specificity  | Accuracy | F1-Score  |
    |-------------------|-----------|------------|----------|-----------|
    | **MobileNetV2**   |           |            |          |           |
    | **OPNet**         |           |            |          |           |

*   **Training Curves:**
  
*   **Interpretation:**

### 9. Parameter Studies & Experiments
*   **Comparison Table:**

    | Model            |  Experiment   | Recall    | Specificity  | Accuracy | F1-Score  |
    |------------------|---------------|-----------|------------|----------|-----------|
    | **MobileNetV2**  | Baseline      |           |            |          |           |
    | **OPNet**        | Baseline      |           |            |          |           |
    | **MobileNetV2**  | 1 desc.       |           |            |          |           |
    | **OPNet**        | 1 desc.       |           |            |          |           |
    | **MobileNetV2**  | 2 desc.       |           |            |          |           |
    | **OPNet**        | 2 desc.       |           |            |          |           |
    | **MobileNetV2**  | 3 desc.       |           |            |          |           |
    | **OPNet**        | 3 desc.       |           |            |          |           |
    | **MobileNetV2**  | 4 desc.       |           |            |          |           |
    | **OPNet**        | 4 desc.       |           |            |          |           |

    
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
