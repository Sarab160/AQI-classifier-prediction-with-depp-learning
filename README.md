# 🌍 **Air Pollution AQI Prediction using ANN**

## 📌 **Project Overview**

This project predicts the **AQI Category** using air pollution data with an Artificial Neural Network (ANN).
The model uses key pollutant values to classify air quality levels.

---

## 📊 **Dataset**

The dataset contains:

* CO AQI Value
* Ozone AQI Value
* NO2 AQI Value
* PM2.5 AQI Value
* AQI Category (Target)

---

## ⚙️ **Steps Performed**

### **1. Data Preprocessing**

* Removed outliers using IQR method (PM2.5 AQI Value)
* Dropped missing values
* Selected important features
* Applied feature scaling using StandardScaler

---

### **2. Feature Selection**

Used only numerical pollutant values:

* CO AQI Value
* Ozone AQI Value
* NO2 AQI Value
* PM2.5 AQI Value

---

### **3. Target Encoding**

* AQI Category encoded using LabelEncoder

---

### **4. Model Architecture**

Artificial Neural Network (ANN):

* Input Layer: 4 features
* Hidden Layers: 4 layers with ReLU activation
* Output Layer: Softmax activation (multi-class classification)

---

## 🧠 **Model Details**

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metrics: Accuracy

---

## 📈 **Results**

* Training Accuracy: ~97%
* Testing Accuracy: ~96%

---

## 🔍 **Key Insight**

The high accuracy is due to the strong relationship between pollutant values and AQI category.
The model effectively learns how pollutant levels determine air quality.

---

## 🚀 **How to Run**

### **1. Install dependencies**

```
pip install pandas seaborn matplotlib scikit-learn tensorflow
```

### **2. Run the script**

```
python main.py
```

