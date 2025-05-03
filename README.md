# 🩺 MediForecaster: Multi-Disease Prediction System

An AI-powered web app that predicts multiple diseases based on user symptoms, provides descriptions, recommended medicines, and preventive measures. Built with **Streamlit** and trained on medical datasets.

## 🚀 Features
- 🔍 Symptom-based disease prediction (over 40+ diseases)
- 📖 Disease descriptions
- 💊 Medicine recommendations
- ✅ Preventive measures
- 🎙️ (Optional) Text-to-speech audio output

---

## 📊 Demos (Screenshots)
*(Add screenshots here for better presentation, e.g., `![Screenshot](path/to/image.png)`)*
  
---

## 🗂️ Dataset Sources
- `Training.csv`
- `Testing.csv`
- `Symptom_severity.csv`
- `symptom_precaution.csv`
- `symptom_Description.csv`
- `diseases_and_medicines17.csv`

---

## ⚙️ Tech Stack
- **Frontend & Backend**: Streamlit
- **ML Model**: Random Forest Classifier (via scikit-learn)
- **Data Handling**: Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib
- **Optional Audio**: pyttsx3

---

## 🚀 How to Run Locally

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/medical-multi-disease-prediction.git
    cd medical-multi-disease-prediction
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📁 Project Structure

