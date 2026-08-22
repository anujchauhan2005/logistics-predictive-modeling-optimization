# Logistics Predictive Modeling and Optimization

## Week 4 Internship Task

This project applies predictive modeling to a logistics problem: **predicting delivery time in minutes** and using the predictions to recommend operational optimization strategies.

### Objectives
- Simulate a realistic logistics delivery dataset.
- Predict delivery time using machine learning.
- Compare Linear Regression and Random Forest Regression.
- Evaluate models using MAE, RMSE, and R².
- Apply 5-fold cross-validation and hyperparameter tuning.
- Translate model insights into logistics optimization recommendations.

### Dataset Features
- `distance_km`
- `traffic_level`
- `weather`
- `vehicle_type`
- `order_weight_kg`
- `items_count`
- `warehouse_load`
- `driver_experience_years`
- `hour_of_day`
- `delivery_time_minutes` (target)

### Project Structure
```text
logistics-predictive-modeling-optimization/
├── data/
│   └── logistics_delivery_data.csv
├── logistics_prediction.py
├── Logistics_Predictive_Modeling.ipynb
├── requirements.txt
├── README.md
└── Week_4_Logistics_Predictive_Modeling_and_Optimization.docx
```

### How to Run

```bash
pip install -r requirements.txt
python logistics_prediction.py
```

The script prints MAE, RMSE, R², the best hyperparameters, and top feature importances.

### Optimization Strategies
The predicted delivery time can support:
1. Dynamic driver allocation
2. Peak-hour staffing
3. Route and order prioritization
4. Warehouse workload balancing
5. Vehicle selection
6. Proactive customer ETA communication
7. Cost-aware resource allocation

### Note
The dataset is simulated for academic purposes. Before production use, the model should be retrained and validated using real historical logistics data.
