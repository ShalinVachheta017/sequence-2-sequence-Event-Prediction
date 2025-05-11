
# 🔥 Sequence-to-Sequence Event Prediction

🚀 **An advanced deep learning-based sequence-to-sequence (Seq2Seq) model for event prediction with cross-validation, hyperparameter tuning, and robust evaluation.**

🔗 **GitHub Repository:** [Sequence-to-Sequence Event Prediction](https://github.com/ShalinVachheta017/sequence-2-sequence-Event-Prediction/tree/main)

---

## 📌 **Project Overview**
This repository implements a **sequence-to-sequence (Seq2Seq)** model using **LSTM** networks to predict event sequences based on historical data. It supports:
- **K-Fold Cross-Validation** for model robustness.
- **Hyperparameter tuning** to find the best configuration.
- **Post-training analysis** to select the best model.
- **Retraining & Evaluation** on unseen test data.

---

## 🏗 **Project Structure**
```
📂 Sequence-to-Sequence-Event-Prediction
│── main.py              # Runs the full training pipeline
│── train.py             # Training & validation functions
│── post_analysis.py     # Selects the best model after training
│── retrain.py           # Retrains the best model & evaluates on test set
│── model_definition.py  # LSTM-based Encoder-Decoder model
│── data_loader.py       # Data preprocessing & k-fold dataset preparation
│── metrics.py           # Custom loss & evaluation metrics
│── utils.py             # Utility functions (logging, config handling, plotting)
│── config.yaml          # Configuration settings for training jobs
│── Events.csv           # Dataset file (input)
│── README.md            # Project Documentation
```

---

## ⚙️ **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ShalinVachheta017/sequence-2-sequence-Event-Prediction.git
cd sequence-2-sequence-Event-Prediction
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Ensure Correct Folder Structure**
Ensure that `Events.csv` (dataset file) is present in the root directory.

---

## 🚀 **How to Train the Model**
The training pipeline performs:
- **Data Loading & Preprocessing**
- **K-Fold Cross-Validation**
- **Hyperparameter Optimization**
- **Performance Metrics Computation**

#### **Run the Training Script**
```bash
python main.py
```
This script:
1. Loads data from `Events.csv`
2. Trains the model with multiple hyperparameters (specified in `config.yaml`).
3. Saves training logs and model outputs in `./logs` and `./outputs` directories.

---

## 📊 **Post-Training Analysis**
To analyze the trained models and select the best-performing one:
```bash
python post_analysis.py
```
This script selects the best model based on **validation coverage and accuracy** while avoiding overfitting.

---

## 🔄 **Retraining and Final Evaluation**
Once the best model is selected, it can be retrained multiple times and evaluated on test data.

```bash
python retrain.py
```
This script:
1. Loads the **best model configuration**.
2. Retrains it multiple times for robustness.
3. Evaluates on the test set.
4. Saves test performance results in `final_test_results.csv`.

---

## 🛠 **Customization**
To modify the training configurations (batch size, learning rate, epochs, etc.), edit **`config.yaml`**:
```yaml
general:
  batch_size: 8
  epochs_list: [40, 80, 100, 120]
  learning_rates: [0.001]
  hidden_sizes: [64, 128, 256]
  num_layers_options: [1, 2, 3]
  bidirectional_options: [true, false]
```

---

## 📈 **Performance Metrics**
The model is evaluated using:
- **Cross-Entropy Loss** for training and validation.
- **Accuracy & Coverage** for post-training and evaluation.

The results are stored in:
- `all_jobs_results.csv` (training phase)
- `final_test_results.csv` (final evaluation)

---

## 🤖 **Model Architecture**
The model is a **Bidirectional LSTM-based Encoder-Decoder**:
```
Encoder (LSTM) → Decoder (LSTM) → Fully Connected Layer
```
The decoder uses **teacher forcing** with a ratio of `0.3` during training.

---

## 🏆 **Results & Visualization**
- Training loss curves, validation loss curves, and accuracy  are saved in `./outputs/`

---

## 📬 **Contributing**
Feel free to **fork**, **open issues**, or submit **pull requests**! 😊

---

## 📝 **License**
This project is open-source under the **MIT License**.

