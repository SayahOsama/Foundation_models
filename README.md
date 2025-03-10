# Chronos Time Series Forecasting

This project uses **AutoGluon Chronos** for **time series forecasting**. It provides a pipeline for **training models**, making **predictions**, and **evaluating performance**.

## 🚀 How to Run the Project

### **1️⃣ Clone the Repository**
If you haven’t cloned the repository, do this:
```sh
git clone https://github.com/SayahOsama/Foundation_models.git
cd Foundation_models
```

### **2️⃣ Install Dependencies**
Create a virtual environment (optional but recommended):
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```
Then install dependencies:
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Script**
To train the model and generate predictions, run:
```sh
python chronos_example.py
```

### **4️⃣ View Results**
After running, check the `results/` folder:
```sh
ls -lh results/
```
You should see:
```
results/
├── output.log           # Execution logs
├── models_results.txt          # Forecasted results
├── zero_shot_predictions_plot.png # Visualization of predictions
```
- **Check logs**: `cat results/output.log`
- **Check results**: `cat results/models_results.txt`
- **Open graph (Linux)**: `xdg-open results/zero_shot_predictions_plot.png`
- **Open graph (Windows)**: `start results/zero_shot_predictions_plot.png`

---

## 📊 **Expected Output**
1. **Model leaderboard stored in** `models.txt`:
    ```
    model                          score_test
    ChronosFineTuned[bolt_small]    -0.791
    ChronosZeroShot[bolt_small]     -0.812
    ```

2. **A graph (`predictions_plot.png`) showing time series forecasts**.

---


## **📌 Notes**
- This project uses **AutoGluon Chronos** for time series forecasting.
- Models are **automatically trained and fine-tuned**.
- Results and logs are stored in the **`results/` folder**.
