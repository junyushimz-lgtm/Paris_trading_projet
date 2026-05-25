import os
import pandas as pd

from src.config import (
    TICKERS,
    START_DATE,
    END_DATE,
    CORR_THRESHOLD,
    PVALUE_THRESHOLD,
    DISTANCE_THRESHOLD,
    STOCK_A,
    STOCK_B,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD
)

from src.data import download_prices

from src.clustering import (
    compute_clusters,
    find_correlated_pairs
)

from src.cointegration import (
    test_cointegration,
    select_best_pairs
)

from src.stationarity_tests import (
    test_spread_stationarity,
    adf_rolling
)

from src.pairs_trading import (
    generate_trading_signals,
    run_simple_backtest,
    compute_performance_metrics,
    optimize_thresholds
)

from src.transaction_costs import (
    TransactionCostModel,
    apply_costs_to_backtest,
    compute_breakeven_cost,
    cost_sensitivity_analysis
)

from src.ml_models import (
    train_classifier,
    train_regressor,
    train_xgboost_classifier
)

from src.ml_no_leakage import (
    train_classifier_no_leakage,
    train_regressor_no_leakage
)

from src.sentiment import compute_sentiment

from src.benchmark import (
    build_benchmark_df,
    compute_strategy_performance,
    apply_ml_filter,
    compute_ml_filtered_performance
)

from src.meta_model import (
    train_meta_model,
    build_final_decision_table
)

from src.reporting import build_summary_results

from src.aggregation import build_final_recommendation

from src.visualization import (
    plot_dendrogram,
    plot_best_pair,
    plot_clusters_pca,
    plot_spread,
    plot_zscore,
    plot_cumulative_return,
    plot_regression_predictions,
    plot_optimized_backtest,
    plot_optimized_zscore,
    plot_benchmark_comparison,
    plot_ml_filtered_strategy,
    plot_lstm_training_history,
    plot_lstm_predictions,
    plot_lstm_regression_predictions
)

from src.deep_learning import (
    train_lstm_classifier,
    train_lstm_regressor
)


def main():

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/ml", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("\n1. Download data")

    prices, returns = download_prices(
        TICKERS,
        START_DATE,
        END_DATE
    )

    print(prices.head())
    print("Returns shape:", returns.shape)

    print("\n2. Correlation and clustering")

    cluster_df, corr_matrix, linkage_matrix = compute_clusters(
        returns,
        DISTANCE_THRESHOLD
    )

    corr_matrix.to_csv(
        "outputs/correlation_matrix.csv"
    )

    cluster_df.to_csv(
        "outputs/clusters.csv",
        index=False
    )

    plot_dendrogram(
        linkage_matrix,
        labels=returns.columns,
        save_path="figures/dendrogram.png"
    )

    cluster_pca_df = plot_clusters_pca(
        returns,
        cluster_df,
        save_path="figures/clusters_pca.png"
    )

    cluster_pca_df.to_csv(
        "outputs/clusters_pca.csv",
        index=False
    )

    print(cluster_df)

    print("\n3. Find correlated pairs")

    pairs_df = find_correlated_pairs(
        cluster_df,
        returns,
        CORR_THRESHOLD
    )

    pairs_df.to_csv(
        "outputs/correlated_pairs.csv",
        index=False
    )

    print(pairs_df)

    print("\n4. Cointegration test")

    cointegration_df = test_cointegration(
        pairs_df,
        prices,
        PVALUE_THRESHOLD
    )

    cointegration_df.to_csv(
        "outputs/cointegration_results.csv",
        index=False
    )

    best_pairs = select_best_pairs(
        cointegration_df
    )

    best_pairs.to_csv(
        "outputs/final_selected_pairs.csv",
        index=False
    )

    print(best_pairs)

    if not best_pairs.empty:

        best_pair = best_pairs.iloc[0]

        plot_best_pair(
            prices,
            best_pair["Stock_A"],
            best_pair["Stock_B"],
            save_path="figures/best_pair.png"
        )

    else:
        print("No cointegrated pairs found.")

    print("\n5. Stationarity tests")

    stationarity_results = test_spread_stationarity(
        prices=prices,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        alpha=0.05
    )

    rolling_adf = adf_rolling(
        spread=stationarity_results["Spread"],
        window=252,
        alpha=0.05
    )

    stationarity_results["Details_DF"].to_csv(
        "outputs/stationarity_results.csv",
        index=False
    )

    rolling_adf.to_csv(
        "outputs/rolling_adf.csv",
        index=True
    )

    print("\n6. Simple pair trading")

    signals, beta, ols_model = generate_trading_signals(
        prices=prices,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD
    )

    print("Hedge Ratio:", beta)

    signals = run_simple_backtest(
        signals
    )

    metrics = compute_performance_metrics(
        signals
    )

    print("Total Return:", round(metrics["Total_Return"] * 100, 2), "%")
    print("Sharpe Ratio:", round(metrics["Sharpe_Ratio"], 2))
    print("Max Drawdown:", round(metrics["Max_Drawdown"] * 100, 2), "%")

    signals.to_csv(
        "outputs/simple_pair_trading_signals.csv",
        index=True
    )

    pd.DataFrame([metrics]).to_csv(
        "outputs/simple_pair_trading_metrics.csv",
        index=False
    )

    plot_spread(
        signals["Spread"],
        title=f"Spread {STOCK_A} / {STOCK_B}",
        save_path="figures/spread.png"
    )

    plot_zscore(
        signals["Zscore"],
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        title=f"Z-Score {STOCK_A} / {STOCK_B}",
        save_path="figures/zscore.png"
    )

    plot_cumulative_return(
        signals,
        title=f"Backtest Pair Trading : {STOCK_A} / {STOCK_B}",
        save_path="figures/simple_backtest.png"
    )

    print("\n7. Transaction costs")

    cost_model = TransactionCostModel(
        commission_pct=0.001,
        bid_ask_spread=0.0005,
        slippage_pct=0.0005
    )

    signals_with_costs = apply_costs_to_backtest(
        signals=signals,
        cost_model=cost_model,
        position_col="Position_A",
        return_col="Strategy_Return"
    )

    breakeven = compute_breakeven_cost(
        signals=signals,
        position_col="Position_A",
        return_col="Strategy_Return"
    )

    sensitivity = cost_sensitivity_analysis(
        signals=signals,
        position_col="Position_A",
        return_col="Strategy_Return",
        cost_range=[0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]
    )

    signals_with_costs.to_csv(
        "outputs/signals_with_transaction_costs.csv",
        index=True
    )

    sensitivity.to_csv(
        "outputs/transaction_cost_sensitivity.csv",
        index=False
    )

    pd.DataFrame([{
        "Breakeven_Cost": breakeven
    }]).to_csv(
        "outputs/breakeven_transaction_cost.csv",
        index=False
    )

    print("\n8. Z-score threshold optimization")

    optimization_df, best_result = optimize_thresholds(
        prices=prices,
        stock_a=STOCK_A,
        stock_b=STOCK_B
    )

    best_signals = best_result["Signals"]

    best_entry = best_result["Entry_Threshold"]
    best_exit = best_result["Exit_Threshold"]

    optimization_df.to_csv(
        "outputs/threshold_optimization.csv",
        index=False
    )

    best_signals.to_csv(
        "outputs/optimized_pair_trading_signals.csv",
        index=True
    )

    plot_optimized_backtest(
        best_signals=best_signals,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        best_entry=best_entry,
        best_exit=best_exit,
        save_path="figures/optimized_backtest.png"
    )

    plot_optimized_zscore(
        best_signals=best_signals,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        best_entry=best_entry,
        best_exit=best_exit,
        save_path="figures/optimized_zscore.png"
    )

    print("\n9. Random Forest classification")

    classifier_result_ml = train_classifier(
        signals
    )

    feature_importance_ml = classifier_result_ml[
        "Feature_Importance"
    ]

    feature_importance_ml.to_csv(
        "outputs/ml_feature_importance.csv",
        index=False
    )

    print("Accuracy:", classifier_result_ml["Accuracy"])

    print("\n10. Regression return J+1")

    regressor_result_ml = train_regressor(
        signals
    )

    prediction_df = regressor_result_ml[
        "Prediction_DF"
    ]

    prediction_df.to_csv(
        "outputs/regression_predictions.csv",
        index=True
    )

    plot_regression_predictions(
        prediction_df,
        save_path="figures/regression_predictions.png"
    )

    print("\n11. ML no leakage")

    classifier_result = train_classifier_no_leakage(
        signals=signals,
        test_size=0.2,
        n_splits_cv=5,
        n_estimators=100
    )

    regressor_result = train_regressor_no_leakage(
        signals=signals,
        test_size=0.2,
        n_estimators=100
    )

    classifier_result["Feature_Importance"].to_csv(
        "outputs/ml_feature_importance_no_leakage.csv",
        index=False
    )

    regressor_result["Prediction_DF"].to_csv(
        "outputs/regression_predictions_no_leakage.csv",
        index=True
    )

    print("\n12. XGBoost classification")

    xgb_result = train_xgboost_classifier(
        signals
    )

    xgb_feature_importance = xgb_result[
        "Feature_Importance"
    ]

    xgb_feature_importance.to_csv(
        "outputs/ml/xgboost_feature_importance.csv",
        index=False
    )

    print("XGBoost Accuracy:", xgb_result["Accuracy"])
    
    print("\n13. LSTM Deep Learning classification et Regression")

    lstm_result = train_lstm_classifier(
        signals=signals,
        sequence_length=5,
        test_size=0.2,
        epochs=5,
        batch_size=64,
        hidden_size=8
    )

    print("LSTM Accuracy:", lstm_result["Accuracy"])
    print(lstm_result["Classification_Report"])
    print(
        "Latest LSTM convergence probability:",
        lstm_result["Latest_Probability"]
    )

    lstm_result["Prediction_DF"].to_csv(
        "outputs/ml/lstm_predictions.csv",
        index=True
    )

    plot_lstm_training_history(
        lstm_result,
        save_path="figures/lstm_training_history.png"
    )

    plot_lstm_predictions(
        lstm_result,
        save_path="figures/lstm_predictions.png"
    )

    print("\n14. NLP sentiment analysis")

    news_data, sentiment_by_stock = compute_sentiment()

    news_data.to_csv(
        "outputs/news_sentiment.csv",
        index=False
    )

    sentiment_by_stock.to_csv(
        "outputs/sentiment_by_stock.csv",
        index=False
    )
    
    print("\nLSTM Deep Learning regression J+1")

    lstm_reg_result = train_lstm_regressor(
        signals=signals,
        sequence_length=5,
        test_size=0.2,
        epochs=10,
        batch_size=64,
        hidden_size=8   
    )

    print("LSTM Regression MAE:", round(lstm_reg_result["MAE"], 6))
    print("LSTM Regression RMSE:", round(lstm_reg_result["RMSE"], 6))
    print("LSTM Regression R2:", round(lstm_reg_result["R2"], 4))
    print("Latest LSTM predicted return J+1:", lstm_reg_result["Latest_Prediction"])

    lstm_reg_result["Prediction_DF"].to_csv(
        "outputs/ml/lstm_regression_predictions.csv",
        index=True
    )

    pd.DataFrame([{
        "MAE": lstm_reg_result["MAE"],
        "RMSE": lstm_reg_result["RMSE"],
        "R2": lstm_reg_result["R2"],
        "Latest_Prediction": lstm_reg_result["Latest_Prediction"]
    }]).to_csv(
        "outputs/ml/lstm_regression_metrics.csv",
        index=False
    )

    plot_lstm_regression_predictions(
        lstm_reg_result,
        save_path="figures/lstm_regression_predictions.png"
    )

    print("\n15. Basic final recommendation")

    final_recommendation = build_final_recommendation(
        signals=signals,
        classifier_result=classifier_result,
        sentiment_by_stock=sentiment_by_stock,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        entry_threshold=ENTRY_THRESHOLD
    )

    final_recommendation.to_csv(
        "outputs/final_recommendation.csv",
        index=False
    )

    print("\n16. Benchmark vs Buy & Hold")

    benchmark_df = build_benchmark_df(
        prices=prices,
        best_signals=best_signals,
        stock_a=STOCK_A,
        stock_b=STOCK_B
    )

    performance = compute_strategy_performance(
        benchmark_df
    )

    benchmark_df.to_csv(
        "outputs/benchmark_comparison.csv",
        index=True
    )

    performance.to_csv(
        "outputs/performance_comparison.csv",
        index=True
    )

    plot_benchmark_comparison(
        benchmark_df=benchmark_df,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        save_path="figures/benchmark_comparison.png"
    )

    print("\n17. ML filtered strategy")

    ml_filtered_signals = apply_ml_filter(
        signals=best_signals,
        classifier_result=classifier_result,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        entry_threshold=best_entry,
        proba_threshold=0.55
    )

    ml_metrics = compute_ml_filtered_performance(
        ml_filtered_signals
    )

    ml_filtered_signals.to_csv(
        "outputs/ml_filtered_signals.csv",
        index=True
    )

    pd.DataFrame([ml_metrics]).to_csv(
        "outputs/ml_filtered_performance.csv",
        index=False
    )

    plot_ml_filtered_strategy(
        best_signals=best_signals,
        ml_filtered_signals=ml_filtered_signals,
        save_path="figures/ml_filtered_strategy.png"
    )

    print("\n18. Final reporting table")

    summary_results = build_summary_results(
        best_result=best_result,
        ml_metrics=ml_metrics,
        ml_filtered_signals=ml_filtered_signals
    )

    summary_results.to_csv(
        "outputs/final_summary_results.csv",
        index=False
    )

    print(summary_results)

    print("\n19. Meta-model")

    meta_result = train_meta_model(
        signals=signals,
        classifier_result=classifier_result,
        regressor_result=regressor_result,
        sentiment_by_stock=sentiment_by_stock,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        best_entry=best_entry,
        proba_threshold=0.55
    )

    final_decision_meta, meta_coefficients = build_final_decision_table(
        signals=signals,
        classifier_result=classifier_result,
        regressor_result=regressor_result,
        sentiment_by_stock=sentiment_by_stock,
        meta_result=meta_result,
        stock_a=STOCK_A,
        stock_b=STOCK_B,
        best_entry=best_entry,
        decision_threshold=0.55
    )

    final_decision_meta.to_csv(
        "outputs/final_decision_meta.csv",
        index=False
    )

    meta_coefficients.to_csv(
        "outputs/meta_model_coefficients.csv",
        index=False
    )

    meta_result["Meta_DF"].to_csv(
        "outputs/meta_model_dataset.csv",
        index=True
    )

    print(final_decision_meta)
    print(meta_coefficients)

    print("\nPipeline terminé.")
    print("Résultats sauvegardés dans outputs/")
    print("Graphiques sauvegardés dans figures/")


if __name__ == "__main__":
    main()