# MultiNetTOFAnalysis: Leveraging Time-of-Flight Imaging with Neural Network Analysis for Material Thickness Estimation

## Overview

MultiNetTOFAnalysis is an innovative research framework dedicated to advancing the field of material science and non-invasive measurement techniques through the application of cutting-edge neural network architectures. This project leverages Time-of-Flight (TOF) imaging data to estimate material thickness with unprecedented accuracy and efficiency. By systematically comparing the performance of various neural network models, including Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), InceptionNet, and DenseNet, MultiNetTOFAnalysis aims to identify the most effective strategies for TOF data analysis.

The project focuses on the utilization of multiple neural network architectures to understand their efficacy in analyzing TOF data for the purpose of material thickness estimation. Through rigorous experimentation and comparative analysis, TOFMultiscopeNetAnalysis seeks to provide insights that can be applied across various domains requiring precise material analysis.

## Understanding the Dataset

The dataset comprises measurements from a Time-of-Flight (TOF) camera, utilized for estimating material thickness. Each row in the dataset represents a pixel value corresponding to a specific point on the material being analyzed. The dataset features real and imaginary parts of signals received at eight different frequencies, labeled as `realPart_1` to `realPart_8` and `imaginaryPart_1` to `imaginaryPart_8`, respectively. These components capture the phase shift of the light as it bounces back from the material, providing essential information about the material's surface and its thickness. Additionally, the dataset includes a ninth real part, `realPart_9`, which denotes the angle between the TOF camera and the material's surface. This angle is crucial for correcting measurements for any angular distortion and ensuring accuracy in thickness estimation. The combination of real, imaginary parts, and the angle information provides a comprehensive set of features for analyzing material properties using advanced machine learning techniques, particularly focusing on neural network models designed for 1D data analysis.

## Understanding the Results

The test results for estimating material thickness from Time-of-Flight (TOF) data using different neural network models are quite insightful. DenseNet demonstrated remarkable precision with a Mean Absolute Error (MAE) of about 0.863, closely followed by InceptionNet with an MAE of 0.867. These two models showed they're particularly good at handling this kind of data. The Convolutional Neural Network (CNN) while slightly less accurate, still performed commendably, with an MAE of 0.984. However, the Artificial Neural Network (ANN) lagged with an MAE of 4.004, indicating it might not be the best choice for this specific task without further tuning. These outcomes suggest that more complex networks like DenseNet and InceptionNet are better suited for accurately estimating material thickness from TOF data. View the boxplot for further understanding of the results.

