## 🧠 Multiple Disease Prediction using Machine Learning

This project is a Multiple Disease Prediction System built with Support Vector Machine (SVM) to classify whether a person is likely to have a specific disease. The diseases covered in this project are:

🩸 Diabetes

❤️ Heart Disease

🧩 Parkinson’s Disease

The datasets for this project are sourced from Kaggle.

# 📂 Project Overview

Healthcare applications often require accurate and fast predictions to assist in early diagnosis. This project leverages machine learning to predict three major diseases using structured medical datasets. By training separate SVM classifiers for each disease, the system can provide reliable results on whether a patient is at risk.

# 🚀 Features

Predicts Diabetes, Heart Disease, and Parkinson’s Disease individually.
Uses Support Vector Machine (SVM) for classification.
Datasets are publicly available on Kaggle.
Modular approach – each disease model can work independently.

# 🛠️ Technologies Used

Python 3
Libraries:
numpy, pandas, scikit-learn, streamlit, Pickle (for saving models)

# ⚙️ How It Works

Data Preprocessing:
Handle missing values (if any).
Normalize/scale the dataset for SVM.
Model Training:
Train an SVM classifier for each disease dataset.
Split data into training and testing sets.
Evaluation:
Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
Prediction:
Input patient’s medical data.
Model outputs whether the patient is likely to have the disease.

# 📈 Results

Diabetes Prediction Accuracy: ~77.2%

Heart Disease Prediction Accuracy: ~86%

Parkinson’s Disease Prediction Accuracy: ~87%

# ▶️ How to Run
1. Clone this repository:
   ```
  git clone https://github.comvasuaggarwal0509/multiple-disease-prediction-app.git
  
  cd multiple-disease-prediction```
   
2. Install dependencies:
   ```
   pip install -r requirements.txt```
3. Run Python script
   ```
   streamlit run app.py```

# 🔮 Future Enhancements

Add more diseases to the prediction system.
Deploying on more feasable and scalable servers like Azure or AWS
Updation of dataset and improving efficiency
Improve performance with deep learning models.

# 👩‍💻 Author

Developed by Vasu Aggarwal 

📌 Data Science & Machine Learning Enthusiast
