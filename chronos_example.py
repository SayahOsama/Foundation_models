from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Load the Australian Electricity dataset
train_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/train.csv",
    id_column="item_id",
    timestamp_column="timestamp",
)

# Zero-shot forecasting using Chronos-Bolt
predictor = TimeSeriesPredictor(prediction_length=48).fit(train_data, presets="bolt_base")

# Generate predictions
predictions = predictor.predict(train_data)

# Load test data for visualization
test_data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv",
    id_column="item_id",
    timestamp_column="timestamp",
)

# Plot predictions
predictor.plot(test_data, predictions, max_history_length=200, item_ids=["T000002"])

# Fine-tune Chronos-Bolt
predictor_fine_tuned = TimeSeriesPredictor(prediction_length=48, eval_metric="MASE").fit(
    train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    enable_ensemble=False,
    time_limit=600,
)

# Evaluate the fine-tuned model
predictor_fine_tuned.leaderboard(test_data)

# Load grocery sales dataset for covariate regressor example
train_data_grocery = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/train.csv",
    id_column="item_id",
    timestamp_column="timestamp",
)

# Fit predictor using covariates
predictor_covariates = TimeSeriesPredictor(
    prediction_length=7,
    eval_metric="MASE",
    target="unit_sales",
    known_covariates_names=["scaled_price", "promotion_email", "promotion_homepage"],
).fit(
    train_data_grocery,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},
            {
                "model_path": "bolt_small",
                "covariate_regressor": "CAT",
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"},
            },
        ],
    },
    time_limit=600,
    enable_ensemble=False,
)

# Evaluate the covariate regressor model
test_data_grocery = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
    id_column="item_id",
    timestamp_column="timestamp",
)
predictor_covariates.leaderboard(test_data_grocery)
