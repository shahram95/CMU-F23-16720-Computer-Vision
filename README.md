# 16-720: Introduction to Computer Vision  
**Instructor:** Srinivasa Narasimhan  

## Course Overview  
This course provides an introduction to computer vision, focusing on how images and videos are processed, analyzed, and interpreted to extract information. It covers fundamental concepts in image formation, feature detection, recognition, 3D reconstruction, and motion analysis. Assignments include hands-on coding and theoretical work aimed at applying vision algorithms to solve real-world problems.

---

## Table of Contents  
1. Homework Submissions    
2. CMU Academic Integrity

---

## Homework Submissions  

### Homework #1  
**Submission Date:** September 21, 2023  

#### Topics Covered:  
- Gaussian Filters and Derivatives  
- Laplacian of Gaussian (LoG)  
- Multiscale Feature Extraction  
- K-means Clustering and Word Maps  
- Confusion Matrix Analysis  
- Data Augmentation with Random Erasing  
- Experimentation with Hyperparameters: Filter Scales, Clustering, and Pyramid Matching  

#### Results:  
- Achieved **80.5% accuracy** using data augmentation.  
- With **MobileNetV2**, achieved **97.5% accuracy**.

---

### Homework #2  
**Submission Date:** October 5, 2023  

#### Topics Covered:  
- Homography and Epipolar Geometry  
- Harris and FAST Corner Detection  
- BRIEF Descriptors and Hamming Distance  
- Implementing Feature Matching and RANSAC for Robust Estimation  

#### Key Experiments:  
- Developed a Harry Potter AR overlay using homography.  
- Experimented with FAST and BRIEF on real-time tracking.  
- Motion subtraction using dominant motion estimation.  

---

### Homework #3  
**Submission Date:** October 24, 2023  

#### Topics Covered:  
- Lucas-Kanade Optical Flow  
- Template Correction for Tracking  
- Dominant Motion Subtraction  
- Epipolar Geometry and Essential Matrices  

#### Key Experiments:  
- Implemented inverse compositional tracking for speed optimization.  
- Visualized results for car and girl sequences with/without template correction.  

---

### Homework #4  
**Submission Date:** November 14, 2023  

#### Topics Covered:  
- Fundamental Matrix and Essential Matrix Estimation  
- Eight-Point Algorithm  
- Triangulation for 3D Reconstruction  
- Epipolar Correspondence  

#### Key Results:  
- Recovered the 3D structure of temple points using triangulation.  
- RANSAC implemented to refine fundamental matrix estimation.  

---

### Homework #5  
**Submission Date:** December 1, 2023  

#### Topics Covered:  
- Neural Networks: Backpropagation and Weight Initialization  
- Softmax and Tanh Activation Functions  
- Overfitting Prevention via Regularization  
- Character Detection in Text Parsing  

#### Key Results:  
- Achieved **78.25% accuracy** using a neural network on NIST data with optimized learning rate and batch size.  
- Visualized learned weight patterns after training.

## 6. CMU Academic Integrity

The contents of this repository are provided for educational purposes and to document my solutions to the assignments for the **Computer Vision (16-720)** course at Carnegie Mellon University. Please note the following:

- **Do not copy** any part of this work. The solutions, code, and explanations provided here are my own work, and copying this content for submission in any academic setting is a violation of the Carnegie Mellon University [Academic Integrity Policy](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).
- Sharing, reproducing, or submitting this work as your own in any form is prohibited and may result in severe academic penalties.

Please use this repository responsibly by using it only as a reference for personal learning and understanding the material.
