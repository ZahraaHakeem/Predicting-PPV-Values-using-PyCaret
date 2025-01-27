# Regression  PPV Values using PyCaret

This project demonstrates how to use the PyCaret library to predict the `PPV` value based on other features (`Gender`, `Age`, `Dur`) using machine learning techniques. Below, we provide an explanation of the process, steps to run the code, and justifications for model selection.

---

## 1. Overview
The task is to predict the continuous variable `PPV` (a regression problem) using features like:
- `Gender` (categorical)
- `Age` (numeric)
- `Dur` (numeric)

We utilize the PyCaret regression module for end-to-end machine learning workflows, which simplifies model comparison, evaluation, and deployment.

---

## 2. Requirements
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required libraries: `pycaret`, `pandas`, `openpyxl`, `sklearn`

### Install Dependencies
To install the necessary dependencies, run:
```bash
!pip install pycaret pandas openpyxl
```

---

## 3. Dataset
The dataset file `TG_T_CashValues_Rel.xlsx` contains columns:
- `Gender`: Categorical data (e.g., "Male", "Female")
- `Age`: Numeric data representing age
- `Dur`: Numeric data for duration
- `PPV`: Target variable (value to be predicted)

Ensure the dataset file is in the same directory as the script/notebook.

---

## 4. How to Run the Code
1. **Load Data**:
   The code reads the data from `TG_T_CashValues_Rel.xlsx` using `pandas`. Ensure the dataset file is present in the directory.

2. **Initialize PyCaret Setup**:
   The `setup()` function automatically handles:
   - Missing values
   - Encoding categorical features (e.g., `Gender`)
   - Scaling numeric features (`Age`, `Dur`)
   - Splitting data into training (80%) and testing (20%) sets

3. **Model Comparison**:
   The `compare_models()` function evaluates multiple regression algorithms and selects the best-performing model based on default metrics (e.g., RMSE).

4. **Model Evaluation**:
   The `evaluate_model()` function provides visual and statistical insights into the selected model's performance.

5. **Predictions**:
   Predictions are made on both the test set and new data:
   - **Test Set**: Calculates performance metrics (MAE, RMSE)
   - **New Data**: Generates predictions for unseen data

6. **Save and Load the Model**:
   The model is saved to a file (`TG_T_CashValues_Rel_pipeline`) and can be reloaded for future use without retraining.

---

## 5. Model Justification
### Why PyCaret?
- **Ease of Use**: PyCaret simplifies data preprocessing, feature engineering, and model comparison.
- **Automation**: It automates repetitive tasks like encoding, scaling, and hyperparameter tuning.
- **Model Benchmarking**: Quickly compares multiple machine learning models.

### Selected Model
The best model is automatically selected by `compare_models()` based on performance metrics. The exact model may vary depending on the dataset but could include algorithms like `RandomForestRegressor`, `XGBoost`, or `LinearRegression`.

### Performance Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference between actual and predicted `PPV` values.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared difference.

These metrics help assess the accuracy and robustness of the model.

---

## 6. Code Explanation

### Key Steps
1. **Load Data**:
   ```python
   df = pd.read_excel('TG_T_CashValues_Rel.xlsx')
   df.head()
   ```

2. **Setup Environment**:
   ```python
   s = setup(df, target='PPV', train_size=0.8, session_id=100,
             numeric_features=['Age', 'Dur'],
             categorical_features=['Gender'])
   ```

3. **Compare Models**:
   ```python
   best_model = compare_models()
   print(best_model)
   ```

4. **Evaluate Model**:
   ```python
   evaluate_model(best_model)
   ```

5. **Predict on Test Set**:
   ```python
   test = predict_model(best_model)
   print("MAE: ", mean_absolute_error(test['PPV'], test['prediction_label']))
   print("RMSE: ", mean_squared_error(test['PPV'], test['prediction_label'], squared=False))
   ```

6. **Save and Load Model**:
   ```python
   save_model(best_model, 'TG_T_CashValues_Rel_pipeline')
   loaded_model = load_model('TG_T_CashValues_Rel_pipeline')
   ```

7. **Predict on New Data**:
   ```python
   new_data = pd.DataFrame({
       'Gender': ['Male'],
       'Age': [0],
       'Dur': [15.7]
   })
   predictions = predict_model(loaded_model, data=new_data)
   print(predictions)
   ```

---

## 7. Output
- **Test Set Predictions**:
  The model's predictions on the test set are displayed, along with MAE and RMSE values.

- **New Data Predictions**:
  Example output for new input data (`Gender: Male, Age: 0, Dur: 15.7`):
  ```
    Gender  Age   Dur  PPV_Prediction
  0   Male    0  15.7         12.34
  ```

- **Saved Model**:
  The trained model is saved to `TG_T_CashValues_Rel_pipeline` and can be reloaded for future use.



