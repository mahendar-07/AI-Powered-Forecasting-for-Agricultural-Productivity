# AI-Powered-Forecasting-for-Agricultural-Productivity


Crop Yield Prediction Model
Overview

This project presents a Crop Yield Prediction Model, developed using a Feed-Forward Neural Network. The model predicts crop yield with remarkable accuracy by considering a diverse range of inputs, including soil nutrients, environmental conditions, and economic factors. This model offers a robust solution to revolutionize precision agriculture by providing farmers and stakeholders with valuable insights to optimize their farming practices.

Features

High Accuracy: Predicts crop yield with 97% accuracy.
Holistic Approach: Integrates multiple input variables to capture complex agricultural dynamics.
Sustainable Farming: Helps optimize resource usage, reduce waste, and improve productivity.
Scalability: Easily extendable to include additional variables or integrate with other agricultural systems.
Model Inputs

The model utilizes the following input features:

Soil Nutrients: Nitrogen, Phosphorus, Potassium (NPK levels)
Environmental Factors: Temperature, Humidity, Rainfall, and pH levels
Applications

Precision Agriculture: Enables farmers to forecast crop yield with high reliability.
Resource Management: Optimizes fertilizer use and water management strategies.
Economic Planning: Supports data-driven decision-making for market readiness.
Future Enhancements

Dynamic Weather Integration: Incorporate real-time weather data.
Geospatial Analysis: Include GPS-based data for field-specific predictions.
Visualization Tools: Develop dashboards to present predictions interactively.
Conclusion

The Crop Yield Prediction Model is a significant advancement in leveraging Artificial Intelligence for agriculture. By providing accurate predictions, it empowers sustainable farming practices and resource optimization, ultimately benefiting farmers and the agricultural community.

AI-Powered Forecasting for Agricultural Productivity uses a Feed-Forward Neural Network (FNN) to predict crop yields with 97% accuracy. Built with TensorFlow, PyTorch, and Flask, it analyzes soil and environmental data to provide real-time crop recommendations, enhancing decision-making and boosting agricultural efficiency.

#SYSTEM WORKFLOW
# System Workflow

```text
                 AI-Powered Crop Recommendation System

┌──────────────────────────────────────────────────────┐
│              Crop Recommendation Dataset             │
│      (2200 Records, 7 Features, 22 Crop Classes)     │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│            Data Cleaning & Preprocessing             │
│  • Check Missing Values                              │
│  • Feature Selection                                 │
│  • Label Encoding                                    │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│               Train-Test Split (70:30)               │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                 Feature Scaling                      │
│        StandardScaler (Fit on Train Data)            │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│          Feed Forward Neural Network (FNN)           │
│                                                      │
│  Input Layer (7 Features)                            │
│          ↓                                           │
│  Dense Layer (256, tanh)                             │
│          ↓                                           │
│  Dense Layer (128, tanh)                             │
│          ↓                                           │
│  Output Layer (22, softmax)                          │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                 Model Training                       │
│  • Adam Optimizer                                    │
│  • Batch Size = 32                                   │
│  • Epochs = 20                                       │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                 Model Evaluation                     │
│  • Training Accuracy ≈ 97%                           │
│  • Testing Accuracy ≈ 96%                            │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                Model Persistence                     │
│  • crop_model.keras                                  │
│  • scaler.pkl                                        │
│  • label_encoder.pkl                                 │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│               Flask Web Application                  │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                 User Input                           │
│  N, P, K, Temperature, Humidity, pH, Rainfall        │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│             Data Standardization                     │
│         Using Saved StandardScaler                   │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│               Crop Prediction                        │
│          Using Trained FNN Model                     │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│              Label Decoding                          │
│      Convert Class Index → Crop Name                 │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│            Recommended Crop Output                   │
│    Rice | Cotton | Banana | Mango | Coffee ...       │
└──────────────────────────────────────────────────────┘
```
# Tech Stack
Programming Language:
    Python(3.12.6)

Machine Learning & Deep Learning :
    TensorFlow / Keras, Scikit-Learn,Data Processing,Pandas,NumPy

Data Visualization:
    Matplotlib,Seaborn

Web Framework:
    Flask

Model Persistence:
    Pickle (scaler.pkl, label_encoder.pkl),Keras Model (crop_model.keras)

Development Tools:
    VS Code,Git,GitHub
    
Dataset:
Crop Recommendation Dataset (2200 records, 22 crop classes)