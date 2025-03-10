from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
import pandas as pd
# Open a file to save outputs
with open("./results/moldels_results.txt", "w") as file:

    # Load the Australian Electricity dataset
    train_data = TimeSeriesDataFrame.from_path(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/train.csv",
        id_column="item_id",
        timestamp_column="timestamp",
    )

    file.write("✅ Loaded training data.\n")

    # Zero-shot forecasting using Chronos-Bolt
    predictor = TimeSeriesPredictor(prediction_length=48).fit(train_data, presets="bolt_base")

    file.write("✅ Predictor trained using Chronos-Bolt.\n")

    # Generate predictions
    predictions = predictor.predict(train_data)

    # Load test data for evaluation
    test_data = TimeSeriesDataFrame.from_path(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv",
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Plot predictions
    predictor.plot(test_data, predictions, max_history_length=200, item_ids=["T000002"])
    plt.savefig("./results/zero_shot_predictions_plot.png")  # Save the plot as an image
    
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

    file.write("✅ Predictor fine-tuned.\n")

    # Evaluate the fine-tuned model
    leaderboard_fine_tuned = predictor_fine_tuned.leaderboard(test_data, silent=True)
    
    # Ensure all columns are printed
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 1000)  # Prevent line breaks
    file.write(f"🏆 Fine-Tuned Model Leaderboard:\n{leaderboard_fine_tuned.to_string(index=False)}\n\n")
    # file.write(f"🏆 Fine-Tuned Model Leaderboard:\n{leaderboard_fine_tuned}\n\n")

    # Load grocery sales dataset for covariate regressor example
    train_data_grocery = TimeSeriesDataFrame.from_path(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/train.csv",
        id_column="item_id",
        timestamp_column="timestamp",
    )

    file.write("✅ Grocery sales dataset loaded.\n")

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

    file.write("✅ Covariate regressor model trained.\n")

    # Evaluate the covariate regressor model
    test_data_grocery = TimeSeriesDataFrame.from_path(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/grocery_sales/test.csv",
        id_column="item_id",
        timestamp_column="timestamp",
    )
    
    leaderboard_covariates = predictor_covariates.leaderboard(test_data_grocery, silent=True)
    file.write(f"📈 Covariate Model Leaderboard:\n{leaderboard_covariates}\n")

    file.write("\n🎉 Done! All results have been saved.\n")
