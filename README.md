# Automated Plant Disease Detection using Vision Transformers (ViT)

## 1. Executive Summary
This project implements a state-of-the-art Deep Learning solution for the automated identification of plant diseases. By leveraging the Vision Transformer (ViT) architecture and high-performance computing (Dual T4 GPUs), the system achieves an overall classification accuracy of 99% across 38 distinct classes of plant leaves. 

## 2. Technical Methodology
The development pipeline utilized a transfer learning approach to maximize accuracy while minimizing training time.

* **Architecture:** The model uses `vit_tiny_patch16_224`, a Transformer-based vision model that utilizes self-attention mechanisms to capture global dependencies within image patches.
* **Hardware Optimization:** To handle the large-scale dataset, the project employed Data Parallelism across two NVIDIA T4 GPUs, significantly increasing training throughput.
* **Data Pipeline:** Images were resized to 224x224 pixels and normalized according to ImageNet standards. Data augmentation (Random Horizontal Flip) was applied to enhance the model's ability to generalize to different leaf orientations.
* **Training Parameters:** The model was optimized using the Adam optimizer with a learning rate of 0.0001 and a Cross-Entropy loss function.



## 3. Dataset Description
The system was trained and validated on the New Plant Diseases Dataset, which contains augmented images of healthy and diseased crop leaves.
* **Total Classes:** 38 (including Apple, Corn, Grape, Tomato, etc.).
* **Validation Samples:** 17,572 images were used to verify model performance.

## 4. Results and Performance Analysis
The model demonstrated exceptional performance, reaching near-perfect metrics across nearly every category.

### 4.1 Key Performance Indicators (KPIs)
* **Global Accuracy:** 99%
* **Macro Average F1-Score:** 0.99
* **Weighted Average F1-Score:** 0.99

### 4.2 Class-Specific Performance (Sample Extract)
| Plant Category | Disease State | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Apple** | Apple Scab | 1.00 | 1.00 | 1.00 |
| **Corn** | Common Rust | 1.00 | 1.00 | 1.00 |
| **Tomato** | Target Spot | 0.98 | 0.96 | 0.97 |
| **Grape** | Black Rot | 1.00 | 1.00 | 1.00 |
| **Tomato** | Yellow Leaf Curl Virus | 1.00 | 1.00 | 1.00 |

*Full results can be found in `final_accuracy_report.txt`.*



### 4.3 Training Progress
* The training loss decreased rapidly from 0.1446 in the first epoch to 0.0199 by the third epoch, indicating efficient convergence.

## 5. Artifacts Generated
* **Model Weights:** `new_plant_diseases.pth`
* **Statistical Report:** `final_accuracy_report.txt`
* **Visual Evidence:** `final_confusion_matrix.png`

## 6. Conclusion
The implementation of the Vision Transformer architecture proved highly successful. The model is capable of distinguishing between very similar disease patterns (e.g., different types of Tomato blights) with high reliability, reaching an accuracy level suitable for agricultural deployment.
