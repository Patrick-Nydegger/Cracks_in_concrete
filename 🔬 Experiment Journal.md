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

```
🎯 Zielsetzung und Prozess
Eine Drohne fliegt einen vordefinierten Kurs entlang eines Objekts (z. B. Brückenpfeiler). Sie nimmt dabei kontinuierlich Bilder auf. Das Deep-Learning-Modell verarbeitet diese Bilder automatisch in Echtzeit oder nach dem Flug und digitalisiert die erkannten Risse (Markierung, Speicherung der GPS-Position und der Bilddatei). Das Ziel ist eine vollständige, lückenlose Erfassung aller Schäden.

❗️ Konsequenz von Fehlern (Risikobewertung)
Falsch-Negativ (FN - Echter Riss wird übersehen): Kritischer Fehler. Dies bedeutet, dass ein potenziell strukturell gefährlicher Riss nicht dokumentiert wird und unbehandelt bleibt. Die Konsequenz ist ein hohes Sicherheitsrisiko.

Falsch-Positiv (FP - Kein Riss wird als Riss markiert): Unkritischer Fehler. Dies führt lediglich zu einer unnötigen manuellen Nachkontrolle an dieser Stelle. Die Konsequenz sind höhere Betriebskosten, aber kein Sicherheitsrisiko.

📊 Empfohlene Metrik
Primäre Metrik: Sensitivity (Recall/Trefferquote)

Begründung: Wir müssen die Anzahl der Falsch-Negativen (FN) minimieren. Die Sensitivität beantwortet die Frage: "Von allen tatsächlichen Rissen, wie viele hat das Modell gefunden?" Hier ist es akzeptabel, einen niedrigeren Schwellenwert zu wählen, um die Wahrscheinlichkeit zu maximieren, jeden Riss zu finden.

Sekundäre Metrik: Precision (Präzision), um zu gewährleisten, dass der Workflow durch zu viele unnötige Kontrollpunkte nicht überlastet wird.

🧑‍💻 Anwendungsfall 2: Manuelle Bildkontrolle / Qualitätssicherung
🎯 Zielsetzung und Prozess
Ingenieure oder Techniker erstellen manuell eine Auswahl von Bildern von verdächtigen Stellen. Das Modell wird als Unterstützung oder zweite Meinung eingesetzt, um schnell zu entscheiden, ob ein Bild zur weiteren Detailanalyse an einen Sachverständigen weitergeleitet werden muss ("Hat dieses Bild einen Riss: Ja/Nein?"). Die Zuverlässigkeit der Klassifikation steht im Vordergrund.

❗️ Konsequenz von Fehlern (Risikobewertung)
Falsch-Negativ (FN - Echter Riss wird übersehen): Mittlerer Fehler. Da die manuelle Auswahl bereits eine Verdachtsfläche war, ist das Risiko geringer als bei der Drohne, aber immer noch unerwünscht.

Falsch-Positiv (FP - Kein Riss wird als Riss markiert): Kritischer Fehler. Da jedes als positiv markierte Bild zu einer teuren, zeitaufwändigen Detailanalyse durch einen hoch bezahlten Experten führt, müssen Falsch-Positive minimiert werden.

📊 Empfohlene Metrik
Primäre Metrik: Precision (Präzision)

Begründung: Wir müssen die Anzahl der Falsch-Positiven (FP) minimieren. Die Präzision beantwortet die Frage: "Von allen Bildern, die das Modell als Riss erkannt hat, wie viele waren tatsächlich Risse?" Hier wählen wir einen höheren Schwellenwert, um sicherzustellen, dass jede Meldung des Modells sehr zuverlässig ist.

Sekundäre Metrik: Sensitivity (Recall), um zu verhindern, dass das Modell zwar präzise, aber nutzlos wird, weil es fast gar keine Risse meldet.
```
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
*   **Primary Metric:**
*   **Justification:**
*   **Secondary Metrics:**
    *   Accuracy:
    *   Sensitivity (Recall):
    *   Specificity:
    *   Precision:

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

    | ID  | Model         | Accuracy | F1-Score | Recall | Precision |
    |-----|---------------|----------|----------|--------|-----------|
    | 001 | **Baseline**  |          |          |        |           |
    | 002 | **Custom CNN**|          |          |        |           |
    | 003 |               |          |          |        |           |

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
