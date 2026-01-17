# ♻️ Space Debris Recycling Score Platform

## 📌 Overview

The **Space Debris Recycling Score Platform** is a data-driven project that evaluates the **recyclability potential of space debris** based on its physical and orbital characteristics.
With the rapid increase of defunct satellites and debris in Earth’s orbit, sustainable space operations require intelligent methods to identify debris that can be **recovered, recycled, or repurposed**.

This project combines **data preprocessing, machine learning, and an interactive web interface** to compute a *recycling score* for space debris objects, helping researchers and planners make informed decisions.

---

## 🎯 Objectives

* Analyze real-world space debris datasets
* Extract meaningful features related to recyclability
* Predict a **Recycling Score** using a trained ML model
* Provide a **user-friendly interface** to input parameters and view results
* Support sustainable and circular space economy initiatives

---

## 🧠 Key Features

* 📊 Data preprocessing and feature engineering on space debris datasets
* 🤖 Machine Learning model for recyclability score prediction
* 🛰️ Inputs such as:

  * Mass of debris
  * Material type
  * Orbit type
  * Orbital location / distance
* 🌐 Interactive **Streamlit-based web interface**
* 📈 Real-time prediction results

---

## 🗂️ Project Structure

```
DebrisRecyclyeScore/
│
├── data/
│   ├── active_debris_updated.csv
│   ├── finalucsdataset.csv
│
├── backend/
│   ├── model.py
│   ├── preprocessing.py
│   └── utils.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Plotly / Matplotlib

---

## 📊 Dataset Description

The project uses curated datasets containing information about:

* Active and inactive space debris
* Orbital parameters
* Physical properties (mass, material, size)
* Mission and object metadata

The datasets are cleaned and transformed to make them suitable for ML training and prediction.

---

## ⚙️ How It Works

1. User enters debris parameters through the web interface
2. Input data is preprocessed and normalized
3. Trained ML model predicts a **Recycling Score**
4. The score indicates the feasibility of debris recycling or recovery

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/kavyapoddar13/DebrisRecyclyeScore.git
cd DebrisRecyclyeScore
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
streamlit run streamlit_app.py
```

---

## 📈 Output

* Displays a **Recycling Score** for the given debris
* Helps identify debris suitable for recycling missions
* Can be extended for mission planning and orbital cleanup analysis

---

## 🌍 Applications

* Space sustainability research
* Orbital debris mitigation planning
* Academic and research projects
* Future on-orbit recycling mission simulations

---

## 🔮 Future Enhancements

* Integration with live orbital tracking APIs
* Advanced deep learning models
* Risk assessment for debris capture
* Visualization of debris location in orbit
* Multi-user authentication and dashboards