# 🪐 Exoplanet Classification with 1D CNN (Kepler Dataset)

This project uses NASA's Kepler exoplanet dataset to classify celestial objects as either **confirmed exoplanets** or **false positives** or **candidate** using a **1D Convolutional Neural Network (CNN)**. The pipeline includes data cleaning, feature selection, scaling, label encoding, and deep learning model training.

---

## 📁 Dataset

**Source:** [Kepler Data on NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

**File Path:**
data/q1_q17_dr25_koi_2025.05.14_22.18.22.csv


**Selected Features:**
- `koi_disposition` (Label: `CONFIRMED` / `FALSE POSITIVE`)
- `koi_period`
- `koi_impact`
- `koi_duration`
- `koi_depth`
- `koi_prad`
- `koi_teq`
- `koi_insol`
- `koi_model_snr`
- `koi_steff`
- `koi_slogg`
- `koi_srad`

---

## ⚙️ Workflow

### 1. Load & Inspect Data
- Load CSV using `pandas`
- Explore and select relevant features

### 2. Data Cleaning
- Drop rows with missing values

### 3. Preprocessing
- Scale features using `MinMaxScaler`
- Encode labels using `LabelEncoder` and convert to one-hot vectors
- Reshape input for 1D CNN: `(samples, features, 1)`

### 4. Modeling
- Build a 1D CNN using **TensorFlow/Keras**
- Train to classify: exoplanet vs false positive

---

## 🧠 Model Input

- **Input Shape:** `(N_samples, 11, 1)`
- **Output:** Softmax activation with 3 neurons  
  (`CONFIRMED` / `FALSE POSITIVE` / `CANDIDATE`)

---

## 🧪 Dependencies

Install required packages:

```bash
pip install pandas scikit-learn tensorflow matplotlib
```
## 🏁 How to Run

- Upload the dataset to your environment

- Run the Jupyter notebook cells sequentially

- Monitor training and evaluate performance on the test set

## 📊 Evaluation Metrics

- Accuracy

- Loss

- (Optional) Confusion Matrix, Precision, Recall

## ROC Curve

![ROC Curve](output.png)

The ROC curve shows that the model performs exceptionally well, with an AUC of 0.95, indicating strong capability in distinguishing between classes. The curve stays close to the top-left corner, reflecting a high true positive rate and a low false positive rate. In contrast, a diagonal line would represent random guessing (AUC = 0.5), which this model clearly surpasses.

