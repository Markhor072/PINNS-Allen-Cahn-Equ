# Human Activity Recognition using Smartphone Accelerometer Data

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

A machine learning project that classifies human physical activities (Walking, Sitting, Standing, etc.) from smartphone sensor data using tri-axial accelerometer and gyroscope measurements.

---

## 📖 Project Overview

Human Activity Recognition (HAR) is a key problem in the field of wearable computing and context-aware systems. This project implements a complete machine learning pipeline to process raw sensor data, extract meaningful features, and train classification models to accurately identify user activities. The system demonstrates the potential for applications in healthcare, fitness tracking, and smart environments.

## 📊 Dataset

The project uses the **UCI HAR Smartphone Dataset**, a well-known benchmark dataset for activity recognition.

- **Source**: [UCI Machine Learning Repository: HAR Smartphone Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Sensors**: Accelerometer & Gyroscope (sampled at 50Hz)
- **Activities**: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
- **Subjects**: 30 volunteers
- **Data**: Pre-processed into 561-feature vectors with time and frequency domain variables.

## 🛠️ Tech Stack & Libraries

- **Programming Language**: Python 3.x
- **Data Handling**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook

## 🚀 Methodology

The project follows a standard ML workflow:

1.  **Data Acquisition & Exploration**: Load and understand the structure of the train/test datasets.
2.  **Data Preprocessing**: Handle missing values, normalize features, and prepare data for modeling.
3.  **Feature Engineering**: Utilize the provided 561 features derived from raw sensor signals, including:
    - **Time-domain**: Mean, standard deviation, entropy, etc.
    - **Frequency-domain**: Fast Fourier Transform (FFT) coefficients.
4.  **Model Training**: Train and evaluate multiple classifiers:
    - Random Forest
    - Gradient Boosting
    - Support Vector Machine (SVM)
    - Logistic Regression
5.  **Model Evaluation**: Compare models based on accuracy, precision, recall, and F1-score. Analyze the confusion matrix to understand per-class performance.
6.  **Results & Visualization**: Generate plots to visualize model performance and feature importance.

## 📈 Results

The trained models achieved high accuracy in classifying the six activities. The **Random Forest** classifier consistently performed among the best, leveraging its ability to handle high-dimensional data well.

| Model                | Accuracy | Precision | Recall | F1-Score |
| -------------------- | :------: | :-------: | :----: | :------: |
| Random Forest        |   **~96%**   |    ~96%   |  ~96%  |   ~96%   |
| Gradient Boosting    |   ~94%   |    ~94%   |  ~94%  |   ~94%   |
| Support Vector Machine |   ~92%   |    ~92%   |  ~92%  |   ~92%   |

*Results are approximate and may vary based on test-train split and hyperparameters.*

## 📁 Repository Structure
├── data/ # Directory for dataset (not included in repo)

├── notebooks/
│ └── HAR_Analysis.ipynb # Main Jupyter Notebook for the entire analysis

├── models/ # (Optional) Saved trained models

├── images/ # Plots and visualizations

├── README.md

└── requirements.txt # Python dependencies


## 🔧 Installation & Execution

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Markhor072/Human-Activity-Recognition-using-Smartphone-Accelerometer-Data.git
    cd Human-Activity-Recognition-using-Smartphone-Accelerometer-Data
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Dataset**
    - Download the UCI HAR Dataset from [this link](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip).
    - Unzip it and place the `UCI HAR Dataset` folder in the project's `data/` directory.

4.  **Run the Jupyter Notebook**
    ```bash
    jupyter notebook
    ```
    Open and run the `notebooks/HAR_Analysis.ipynb` notebook step-by-step.

## 👨‍💻 Author

**Shahid Hassan**

- GitHub: [@Markhor072](https://github.com/Markhor072)
- LinkedIn: [Shahid Hassan](https://www.linkedin.com/in/markhor072)
- Portfolio: [https://shahidhassan.vercel.app](https://shahidhassan.vercel.app)

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

- The researchers who created and made the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) publicly available.
- The open-source community for maintaining the invaluable Python data science libraries.
