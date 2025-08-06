# Multiple Disease Prediction System

This project is a web-based application that predicts the likelihood of a user having certain diseases, such as Diabetes and Heart Disease, based on the input of various health-related parameters. This tool is intended to help users get a preliminary idea about their health condition, but it is not a substitute for professional medical advice.

## 📖 Table of Contents
* [About the Project](#about-the-project)
* [Features](#features)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Challenges & Methodology](#-challenges--methodology)
* [Model Performance](#-model-performance)
* [Technologies Used](#️-technologies-used)
* [Datasets](#-datasets)
* [Contributing](#-contributing)
* [License](#-license)
* [Contact](#-contact)


## 📝 About the Project

The Multiple Disease Prediction System is a machine learning project that aims to provide a simple and intuitive interface for users to check for the possibility of having Diabetes or Heart Disease. The predictions are made using machine learning models that have been trained on relevant datasets. The application is built using Streamlit, a popular Python framework for building interactive web applications for machine learning and data science projects.

## ✨ Features

* **Diabetes Prediction:** Predicts the likelihood of a person having diabetes based on features like Glucose level, Blood Pressure, BMI, etc.
* **Heart Disease Prediction:** Predicts the likelihood of a person having heart disease based on features like age, sex, cholesterol levels, etc.
* **User-friendly Interface:** A simple and easy-to-use interface for users to input their data and get predictions.
* **Instantaneous Results:** The prediction results are displayed instantly on the web interface.

## 🚀 Getting Started

To get a local copy up and running follow these simple example steps.

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
    *(**Note:** You will need to create a `requirements.txt` file that lists all the libraries your project uses, such as pandas, scikit-learn, streamlit, etc.)*

##  Usage

1.  **Run the Streamlit application**
    ```sh
    streamlit run app.py
    ```
    *(Assuming your main application file is named `app.py`)*

2.  **Open the application in your browser**
    * The application will open in your default web browser. You can access it at `http://localhost:8501`.

3.  **Select a disease from the sidebar**
    * Choose between Diabetes and Heart Disease prediction.

4.  **Enter the required parameters**
    * Fill in the form with the necessary health-related information.

5.  **Click on the 'Predict' button**
    * The prediction result will be displayed on the screen.

##  Challenges & Methodology

### Exploratory Data Analysis (EDA) & Feature Engineering

Working with medical datasets presents unique challenges. Here are some of the key hurdles encountered and how they were addressed during the EDA and feature engineering phase:

* **Handling Missing or Zero Values:** A significant challenge was dealing with zero values in columns where they are biologically impossible, such as `Glucose`, `BloodPressure`, or `BMI`. These were not truly zero but likely missing data points. To preserve the dataset's integrity, these zero values were imputed with the mean or median of the respective feature.
* **Imbalanced Datasets:** The datasets for both diseases were imbalanced, with a significantly higher number of non-disease instances than disease instances. This can lead to a model that is biased towards the majority class. The performance was evaluated using metrics like F1-score, Precision, and Recall, which are more informative than accuracy in such cases.
* **Outlier Detection:** Medical data often contains outliers (e.g., exceptionally high or low readings). Visualizations like box plots were used to identify these outliers. They were retained in this project to capture the full spectrum of data, but in a real-world clinical application, these would require verification by a domain expert.
* **Feature Scaling:** The features in the datasets had widely different scales. To ensure that no single feature dominated the learning process, `StandardScaler` from Scikit-learn was used to standardize all features to a common scale. This is a crucial step for many machine learning algorithms to perform correctly.

##  Model Performance

The performance of the trained models was evaluated using various metrics on both the training and testing sets. The results are detailed below.

### Diabetes Prediction Model

The model shows strong performance on the training data. On the testing data, it maintains a very high recall ($0.898$), indicating it is effective at correctly identifying patients with diabetes. The precision of $0.702$ suggests that while it catches most positive cases, it also incorrectly flags some non-diabetic patients.

**Training Set Metrics:**
* **Accuracy:** $0.983$
* **F1-Score:** $0.982$
* **Precision:** $0.969$
* **Recall:** $0.997$
* **Confusion Matrix:**
    ```
    [[69781,  2170],
     [  196, 67807]]
    ```

**Testing Set Metrics:**
* **Accuracy:** $0.967$
* **F1-Score:** $0.788$
* **Precision:** $0.702$
* **Recall:** $0.898$
* **Confusion Matrix:**
    ```
    [[17372,   502],
     [  134,  1185]]
    ```

### Heart Disease Prediction Model

The model achieves perfect scores on the training data, which is a strong indicator of **overfitting**. This means the model has learned the training data too well and may not generalize perfectly to new, unseen data. Despite this, the performance on the testing data is still high, with an accuracy of $0.870$ and a strong F1-Score of $0.885$. Future work could involve using techniques like regularization or cross-validation to create a more generalized model.

**Training Set Metrics:**
* **Accuracy:** $1.0$
* **F1-Score:** $1.0$
* **Precision:** $1.0$
* **Recall:** $1.0$
* **Confusion Matrix:**
    ```
    [[333,   0],
     [  0, 401]]
    ```

**Testing Set Metrics:**
* **Accuracy:** $0.870$
* **F1-Score:** $0.885$
* **Precision:** $0.860$
* **Recall:** $0.911$
* **Confusion Matrix:**
    ```
    [[68, 15],
     [ 9, 92]]
    ```

## 🛠️ Technologies Used

* **Python:** The core programming language used for the project.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For building and training the machine learning models.
* **Streamlit:** For creating the web application and user interface.
* **Jupyter Notebook:** For model development and experimentation.

## 📊 Datasets

* **Diabetes Dataset:** [Link to your dataset or mention the source, e.g., PIMA Indians Diabetes Database]
* **Heart Disease Dataset:** [Link to your dataset or mention the source, e.g., Cleveland Heart Disease dataset from the UCI Machine Learning Repository]

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request


## 📧 Contact

Shantanu Gupta - [shantigupta908@gmail.com](shantigupta908@gmail.com)

Project Link: [https://github.com/Shansgupta/MultipleDiseasePrediction](https://github.com/Shansgupta/MultipleDiseasePrediction)