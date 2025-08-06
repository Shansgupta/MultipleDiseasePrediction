# Multiple Disease Prediction System

[![](https://img.shields.io/badge/Live_Demo-Open_App-blue?style=for-the-badge&logo=render)](https://multiplediseaseprediction-1.onrender.com)

This project is a web-based application that predicts the likelihood of a user having certain diseases, such as Diabetes and Heart Disease, based on the input of various health-related parameters. This tool is intended to help users get a preliminary idea about their health condition, but it is not a substitute for professional medical advice.

---
## 📖 Table of Contents
* [Live Demo](#-live-demo)
* [About the Project](#-about-the-project)
* [Features](#-features)
* [Model Input Parameters](#-model-input-parameters)
* [Getting Started](#-getting-started)
* [Usage](#-usage)
* [Challenges & Methodology](#-challenges--methodology)
* [Model Performance](#-model-performance)
* [Technologies Used](#️-technologies-used)
* [Datasets](#-datasets)
* [Contributing](#-contributing)
* [License](#-license)
* [Contact](#-contact)

---
## 🌐 Live Demo

The application has been successfully deployed and is live on Render. You can access and interact with the prediction system directly in your browser.

**[➡️ Try the Live Application Here!](https://multiplediseaseprediction-1.onrender.com)**

---
## 📝 About the Project

The Multiple Disease Prediction System is a machine learning project **built entirely from scratch**. It aims to provide a simple and intuitive interface for users to check for the possibility of having Diabetes or Heart Disease. The entire pipeline, from data preprocessing and feature engineering to model training and deployment, was developed as part of this project. The predictions are made using machine learning models trained on relevant datasets, and the application is built with **Streamlit** to create an interactive user experience.

---
## ✨ Features

* **Live Deployment:** The application is deployed on Render for easy access and demonstration.
* **Diabetes Prediction:** Predicts the likelihood of a person having diabetes.
* **Heart Disease Prediction:** Predicts the likelihood of a person having heart disease.
* **User-friendly Interface:** A simple and easy-to-use interface for users to input their data and get predictions.
* **Instantaneous Results:** The prediction results are displayed instantly on the web interface.

---
## 📋 Model Input Parameters

The models require the following parameters to make a prediction:

### For Diabetes Prediction
* **Pregnancies:** Number of times pregnant
* **Glucose:** Plasma glucose concentration over 2 hours in an oral glucose tolerance test
* **BloodPressure:** Diastolic blood pressure (mm Hg)
* **SkinThickness:** Triceps skin fold thickness (mm)
* **Insulin:** 2-Hour serum insulin (mu U/ml)
* **BMI:** Body mass index (weight in kg/(height in m)^2)
* **DiabetesPedigreeFunction:** A function that scores the likelihood of diabetes based on family history
* **Age:** Age in years

### For Heart Disease Prediction
* **age:** Age of the patient in years
* **sex:** Sex of the patient (1 = male; 0 = female)
* **cp:** Chest pain type (0-3)
* **trestbps:** Resting blood pressure (in mm Hg)
* **chol:** Serum cholesterol in mg/dl
* **fbs:** Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
* **restecg:** Resting electrocardiographic results (0,1,2)
* **thalach:** Maximum heart rate achieved
* **exang:** Exercise induced angina (1 = yes; 0 = no)
* **oldpeak:** ST depression induced by exercise relative to rest
* **slope:** The slope of the peak exercise ST segment
* **ca:** Number of major vessels (0-3) colored by fluoroscopy
* **thal:** Thallium Stress Test result (1 = normal; 2 = fixed defect; 3 = reversable defect)

---
## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Python and pip installed on your system. You can download Python from [here](https://www.python.org/downloads/).

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/Shansgupta/MultipleDiseasePrediction.git](https://github.com/Shansgupta/MultipleDiseasePrediction.git)
    ```
2.  **Navigate to the project directory**
    ```sh
    cd MultipleDiseasePrediction
    ```
3.  **Install the required dependencies**
    ```sh
    pip install -r requirements.txt
    ```
    *(**Note:** You will need to create a `requirements.txt` file that lists all the libraries your project uses.)*

---
## 🎈 Usage

1.  **Run the Streamlit application locally**
    ```sh
    streamlit run app.py
    ```
    *(Assuming your main application file is named `app.py`)*

2.  **Open the application in your browser**
    * The application will open in your default web browser at `http://localhost:8501`.

---
## 🧠 Challenges & Methodology

### Exploratory Data Analysis (EDA) & Feature Engineering

Working with medical datasets presents unique challenges. Here are some of the key hurdles encountered and how they were addressed:

* **Handling Missing or Zero Values:** A significant challenge was dealing with zero values in columns where they are biologically impossible, such as `Glucose`, `BloodPressure`, or `BMI`. These were imputed with the mean or median of the respective feature.
* **Imbalanced Datasets:** The datasets for both diseases were imbalanced. Performance was evaluated using metrics like F1-score, Precision, and Recall, which are more informative than accuracy in such cases.
* **Outlier Detection:** Medical data often contains outliers. Visualizations like box plots were used to identify them. They were retained to capture the full spectrum of data.
* **Feature Scaling:** Features had widely different scales. `StandardScaler` from Scikit-learn was used to standardize all features to a common scale.

---
## 📈 Model Performance

The performance of the trained models was evaluated using various metrics on both the training and testing sets.

### Diabetes Prediction Model

The model shows strong performance on the training data. On the testing data, it maintains a very high recall ($0.898$), indicating it is effective at correctly identifying patients with diabetes. The precision of $0.702$ suggests that while it catches most positive cases, it also incorrectly flags some non-diabetic patients.

**Training Set Metrics:**
* **Accuracy:** $0.983$
* **F1-Score:** $0.982$
* **Precision:** $0.969$
* **Recall:** $0.997$

### Heart Disease Prediction Model

The model achieves perfect scores on the training data, which is a strong indicator of **overfitting**. Despite this, the performance on the testing data is still high, with an accuracy of $0.870$ and a strong F1-Score of $0.885$.

**Training Set Metrics:**
* **Accuracy:** $1.0$
* **F1-Score:** $1.0$
* **Precision:** $1.0$
* **Recall:** $1.0$

---
## 🛠️ Technologies Used

* **Python:** The core programming language used for the project.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For building and training the machine learning models.
* **Streamlit:** For creating the web application and user interface.
* **Render:** For cloud deployment of the application.

---
## 📊 Datasets

* **Diabetes Dataset:** [Link to your dataset or mention the source, e.g., PIMA Indians Diabetes Database]
* **Heart Disease Dataset:** [Link to your dataset or mention the source, e.g., Cleveland Heart Disease dataset from the UCI Machine Learning Repository]

---
## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---
## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
## 📧 Contact

Shantanu Gupta - [shantigupta908@gmail.com](shantigupta908@gmail.com)

Project Link: [https://github.com/Shansgupta/MultipleDiseasePrediction](https://github.com/Shansgupta/MultipleDiseasePrediction)