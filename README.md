# Breast Cancer Prediction System

A machine learning web application that predicts whether a breast tumor is benign or malignant using Support Vector Machine (SVM).

**⚠️ EDUCATIONAL PURPOSE ONLY - NOT A MEDICAL DIAGNOSTIC TOOL**

## 🎯 Project Overview

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Accuracy**: ~96%
- **Features Used**: 5 tumor characteristics from FNA measurements
- **Framework**: Flask
- **Deployment**: Render.com

## 📊 Model Development

### Dataset
- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Total Samples**: 569 tumors
- **Classes**: Benign (357) and Malignant (212)
- **Malignancy Rate**: 37.26%

### Preprocessing Steps

1. **Missing Value Handling**:
   - No missing values in this dataset
   - All features are complete

2. **Feature Selection**:
   - Selected 5 features from 8 available mean features
   - Features chosen based on clinical relevance:
     * mean radius
     * mean texture
     * mean perimeter
     * mean area
     * mean smoothness

3. **Target Variable Encoding**:
   - 0 = Malignant (cancerous)
   - 1 = Benign (non-cancerous)

4. **Feature Scaling** (Mandatory for SVM):
   - StandardScaler applied to all features
   - Mean normalized to ~0
   - Standard deviation normalized to ~1

### Model Training

**Algorithm**: Support Vector Machine (SVM)

**Hyperparameters**:
- Kernel: RBF (Radial Basis Function)
- C: 1.0 (regularization parameter)
- Gamma: 'scale' (kernel coefficient)
- Random state: 42
- Probability: True (for probability estimates)

**Training/Test Split**: 80/20 with stratification

### Model Evaluation
```
Accuracy: 96.49%

Classification Report:
                   precision    recall  f1-score   support
    Malignant         0.97      0.92      0.94        38
    Benign            0.96      0.99      0.97        76

    accuracy                              0.96       114
   macro avg          0.97      0.96      0.96       114
weighted avg          0.96      0.96      0.96       114

Confusion Matrix:
                Predicted
                Mal  Ben
Actual Mal    [ 35    3]
Actual Ben    [  1   75]
```

**Key Metrics**:
- **Accuracy**: 96.49%
- **Precision (Malignant)**: 0.97
- **Recall (Malignant)**: 0.92
- **F1-Score (Malignant)**: 0.94

**Model Interpretation**:
- High precision (97%) for malignant cases means few false positives
- Good recall (92%) means most malignant cases are detected
- Excellent performance on benign cases (99% recall)

### Support Vectors
- Uses ~30% of training samples as support vectors
- Balanced representation from both classes

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
# Clone repository
git clone https://github.com/YourUsername/BreastCancer_Project_YourName_MatricNo.git
cd BreastCancer_Project_YourName_MatricNo

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Usage

1. Open browser: `http://127.0.0.1:5000`
2. Enter tumor feature measurements:
   - Mean Radius (5.0 - 35.0)
   - Mean Texture (5.0 - 45.0)
   - Mean Perimeter (30.0 - 200.0)
   - Mean Area (100.0 - 2500.0)
   - Mean Smoothness (0.05 - 0.20)
3. Click "Analyze Tumor"
4. View prediction and probability

## 🌐 Live Demo

**URL**: https://breast-cancer-predictor-yourname.onrender.com

**Note**: First load may take 30-60 seconds (free tier)

## 📁 Project Structure
```
BreastCancer_Project_YourName_MatricNo/
├── app.py                          # Flask application (production-ready)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── BreastCancer_hosted_webGUI_link.txt  # Deployment info
├── .gitignore                      # Git ignore rules
├── model/
│   ├── model_building.ipynb        # Training notebook with full analysis
│   ├── breast_cancer_model.pkl     # Trained SVM model
│   └── scaler.pkl                  # Feature scaler
├── static/
│   └── style.css                   # Responsive styling
└── templates/
    └── index.html                  # Web interface
```

## 🔧 API Endpoints

### `GET /`
Renders the main web interface

### `POST /predict`
Predicts tumor classification

**Request Body** (form-data):
```
radius_mean: float (5.0 - 35.0)
texture_mean: float (5.0 - 45.0)
perimeter_mean: float (30.0 - 200.0)
area_mean: float (100.0 - 2500.0)
smoothness_mean: float (0.05 - 0.20)
```

**Response** (JSON):
```json
{
  "prediction": 1,
  "prediction_text": "Benign",
  "benign_probability": 95.3,
  "malignant_probability": 4.7,
  "tumor_features": {
    "radius": 12.0,
    "texture": 18.0,
    "perimeter": 80.0,
    "area": 500.0,
    "smoothness": 0.09
  }
}
```

**Error Responses**:
- `400`: Invalid or missing input data
- `500`: Server or model loading error

### `GET /health`
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## 🛠 Technologies Used

- **Backend**: Flask 3.0.0
- **ML Library**: scikit-learn 1.3.0+
- **Data Processing**: pandas 2.1.0+, numpy 1.24.0+
- **Model Persistence**: joblib 1.3.0+
- **Deployment**: Render.com with Gunicorn
- **Algorithm**: Support Vector Machine (SVM)

## 🔒 Production Features

- **Debug Mode Control**: Environment-based toggle (off in production)
- **Robust Path Handling**: Absolute paths anchored to `__file__`
- **Input Validation**: Server-side validation with range checks
- **Error Handling**: Comprehensive try-catch with user-friendly messages
- **Missing Field Protection**: Uses `.get()` for safe form data access
- **Health Monitoring**: `/health` endpoint for deployment checks

## ⚠️ Medical Disclaimer

This system is developed **strictly for educational purposes** to demonstrate machine learning applications in healthcare analytics. 

**This tool must NOT be used for**:
- Actual medical diagnosis
- Clinical decision making
- Patient treatment planning
- Replacing professional medical evaluation

**Always consult qualified healthcare professionals** for any medical concerns or diagnosis.

## 📚 Educational Context

This project demonstrates:
- End-to-end ML pipeline development
- Data preprocessing and feature engineering
- Model training and evaluation
- Web application development
- API design and implementation
- Production deployment practices
- Responsible AI communication

## 👨‍💻 Author

**[Your Name]**  
Matric No: [Your Matric Number]

## 📄 License

This project is for educational purposes only.

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository
- Inspiration: Breast Cancer Wisconsin (Diagnostic) Dataset by Dr. William H. Wolberg