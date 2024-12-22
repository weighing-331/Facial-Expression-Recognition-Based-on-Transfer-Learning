# **Facial Expression Recognition Based on Transfer Learning**

This repository implements a binary classification model for facial expression recognition (Happy vs Sad) using transfer learning. Three models—**VGG16**, **LeNet**, and **SqueezeNet**—were trained and evaluated on the **FER2013** dataset.

---

## **Overview**
Facial expression recognition (FER) plays a crucial role in applications like mental health monitoring, emotion detection, and human-computer interaction. This project explores the use of transfer learning to classify emotions efficiently, leveraging pre-trained deep learning models.

---

## **Dataset**
- **Source**: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Classes Used**: Only "Happy" and "Sad" images from the **test set**.
- **Data Structure**:
  ```
  test/
  ├── happy/
  ├── sad/
  ```

Ensure the dataset is organized in the `test/` directory before running the scripts.

---

## **Project Structure**
```
Facial-Expression-Recognition/
├── Lenet_for.m                # Script for training and evaluating LeNet (big dataset)
├── Squeezenet.m               # Script for general SqueezeNet evaluation
├── Squeezenet_for.m           # Script for SqueezeNet evaluation with specific dataset sizes
├── merchData.m                # Data loading and preprocessing script
├── VGG16.m                    # Script for training and evaluating VGG16
├── lenet_results_big.xlsx     # Results for LeNet (big dataset)
├── lenet_results_small.xlsx   # Results for LeNet (small dataset)
├── model_results_big.xlsx     # Combined results for big dataset
├── model_results_small.xlsx   # Combined results for small dataset
├── VGG16_results.xlsx         # Results for VGG16
├── stat_summary.xlsx          # Summary statistics for all models
├── plot.r                     # R script for generating performance plots
├── test/                      # Dataset folder
│   ├── happy/
│   ├── sad/
├── performance/               # Performance plots
│   ├── Accuracy.png
│   ├── F1 score.png
│   ├── Precision.png
│   ├── Recall.png
│   ├── ROC AUC.png
├── fig/                       # Additional figures
├── README.md                  # Project documentation
```

---

## **Models and Training**
The project evaluates three deep learning models:
1. **VGG16**: A robust model for image classification with transfer learning.
2. **LeNet**: A lightweight model optimized for grayscale images.
3. **SqueezeNet**: A compact and efficient model for low-resource environments.

### **Training Details**:
- **Datasets**:
  - **Big Dataset**: Training with 1000 images, testing with 200 images.
  - **Small Dataset**: Training with 100 images, testing with 20 images.
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Learning Rate**: 0.0001
- **Epochs**: 5, 10, 15, 20

---

## **Results**

| **Model**       | **Dataset** | **Epochs** | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **ROC-AUC** |
|------------------|-------------|------------|--------------|---------------|------------|--------------|-------------|
| **LeNet**        | Big         | 20         | 66.5%        | 69.7%         | 75.9%      | 72.5%        | 71.1%       |
| **LeNet**        | Small       | 20         | 60.0%        | 62.3%         | 78.3%      | 68.3%        | 54.8%       |
| **SqueezeNet**   | Big         | 20         | 85.9%        | 91.3%         | 84.4%      | 87.5%        | 93.1%       |
| **SqueezeNet**   | Small       | 20         | 69.0%        | 78.3%         | 71.7%      | 72.1%        | 75.0%       |

---

## **Performance Visualizations**
Performance plots for accuracy, precision, recall, F1-score, and ROC-AUC are available in the `performance/` folder:
- **Accuracy**: `Accuracy.png`
- **F1-Score**: `F1 score.png`
- **Precision**: `Precision.png`
- **Recall**: `Recall.png`
- **ROC-AUC**: `ROC AUC.png`
<img width="288" alt="ROC AUC" src="https://github.com/user-attachments/assets/54ac984f-6350-4855-8029-8bd0d7c0e925" />
<img width="288" alt="Recall" src="https://github.com/user-attachments/assets/0956eeee-0cfd-4d3f-bf45-213f985ee24f" />
<img width="288" alt="Precision" src="https://github.com/user-attachments/assets/0fd7fb89-a5cb-4a98-af4e-3dc13fe444e0" />
<img width="288" alt="F1 score" src="https://github.com/user-attachments/assets/34c49e06-8cf5-4a58-b267-67fe834e45a0" />
<img width="288" alt="Accuracy" src="https://github.com/user-attachments/assets/c9cc69ca-f042-4b11-97d2-a62b4393e4b4" />

---

## **How to Run**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/weighing-331/Facial-Expression-Recognition-Based-on-Transfer-Learning.git
   cd Facial-Expression-Recognition-Based-on-Transfer-Learning
   ```

2. **Prepare the Dataset**:
   - Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
   - Extract the "Happy" and "Sad" test images into the `test/` directory.

3. **Run the Scripts**:
   - Open MATLAB and execute:
     - `Lenet_for.m` for LeNet (big dataset).
     - `Squeezenet.m` or `Squeezenet_for.m` for SqueezeNet evaluation.
     - `VGG16.m` for VGG16 evaluation.
   - Results and visualizations will be saved in the `results/` and `performance/` folders.

