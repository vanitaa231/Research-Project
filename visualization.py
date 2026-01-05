import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-whitegrid')

# import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

#---------------- Forecast Visualizations--------------------------

def plot_forecast_summary(train, test, hw_forecast, lr_pred, tree_pred, save_path="static/forecast_line.png"):

    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot last 90 days
    recent_train = train[-90:]

    #  Actual and Forecast lines
    plt.plot(recent_train.index, recent_train.values, 
             label="Historical Sales", color="#7f8c8d", linewidth=2, alpha=0.8)
    plt.plot(test.index, test.values, 
             label="Actual (Test Data)", color="black", linewidth=2.2)
    plt.plot(hw_forecast.index, hw_forecast.values, 
             label="Holt-Winters Forecast", color="#e74c3c", linestyle="--", linewidth=2)
    plt.plot(lr_pred.index, lr_pred.values, 
             label="Linear Regression Forecast", color="#3498db", linestyle="-.", linewidth=2)
    plt.plot(tree_pred.index, tree_pred.values, 
             label="Decision Tree Forecast", color="#27ae60", linestyle=":", linewidth=2)

    #  Shaded area for forecast period
    forecast_start = hw_forecast.index[0]
    forecast_end = hw_forecast.index[-1]
    plt.axvspan(forecast_start, forecast_end, color="gray", alpha=0.1, label="Forecast Period")

    #  Highlight max actual sales point
    max_point = test.idxmax()
    plt.annotate("Peak Sales", 
                 xy=(max_point, test[max_point]),
                 xytext=(max_point, test[max_point] + (0.05 * test.max())),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=9, color="black")

    #  Formatting
    plt.title(" Sales Forecast Comparison: Actual vs Models", fontsize=15, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales Amount", fontsize=12)
    plt.grid(alpha=0.3, linestyle="--")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)


    plt.text(0.01, 0.95, "Shaded area indicates forecast period", transform=plt.gca().transAxes,
             fontsize=9, color="gray", style="italic")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast( test, lr_pred, save_path="static/forecasting.png"):
    plt.figure(figsize=(12, 6))

    x = np.arange(len(test))
    bar_width = 0.25

    plt.bar(x - bar_width, test.values, width=bar_width, label="Actual (Test)", color="#9FA2A0")
   
    plt.bar(x + bar_width, lr_pred.values, width=bar_width, label="Linear Regression", color="#25AA4B")
   
    plt.title("Sales Forecast Comparison (Bar Chart)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount")
    plt.xticks(x, test.index.strftime('%b %d'), rotation=45)
    plt.grid(alpha=0.3, axis='y')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Model Comparison
def plot_mape_comparison(results, save_path="static/model_comparison.png"):
    metrics_df = (
        pd.DataFrame(results).T
        .apply(pd.to_numeric, errors="coerce")
        .sort_values("MAPE", ascending=True)
    )
    colors = ["#007BFF", "#FF4D4D", "#33CC33"][:len(metrics_df)]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics_df.index, metrics_df["MAPE"], color=colors, edgecolor="black")
    plt.title("Model Performance (MAPE % Comparison for 30 days test data  )", fontsize=14, fontweight="bold")
    plt.ylabel("MAPE (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2,
                 f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Explainable AI Visualization

def explain_model_simplified(model, X, model_name, save_prefix="static/explain"):
    """Creates explainability ONLY for Linear Regression — Decision Tree explanation removed."""

    # If model is not Linear Regression skip explanation
    if "linear" not in model_name.lower():
        print(f"Skipping explainability: {model_name} is not a Linear model.")
        return False

    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    summary_text_path = os.path.join(os.path.dirname(save_prefix), f"{model_name}_explanation.txt")

    try:

        # Use SHAP if available

        if SHAP_AVAILABLE:
            import shap

            # Masker logic for SHAP
            if hasattr(shap, "maskers") and hasattr(shap.maskers, "Independent"):
                masker = shap.maskers.Independent(X)
                explainer = shap.LinearExplainer(model, masker)
            else:
                explainer = shap.LinearExplainer(model, X)

            shap_values = explainer(X)

            # Compute feature importance
            shap_df = pd.DataFrame({
                "Feature": X.columns,
                "Mean |SHAP value|": np.abs(shap_values.values).mean(axis=0)
            }).sort_values("Mean |SHAP value|", ascending=False)

            # Plot
            plt.figure(figsize=(8, 5))
            plt.barh(shap_df["Feature"], shap_df["Mean |SHAP value|"], color="#FF9933", edgecolor="black")
            plt.gca().invert_yaxis()
            plt.title(f"Top Factors Affecting Sales — {model_name}")
            plt.xlabel("Average Impact on Prediction")
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_{model_name}_simple.png", dpi=300)
            plt.close()

            # Save summary text
            top_features = shap_df.head(3)["Feature"].tolist()
            summary = (
                f"Explainability for {model_name}\n"
                f"Top 3 most influential factors:\n"
                + "\n".join([f"• {f}" for f in top_features])
            )
            with open(summary_text_path, "w") as f:
                f.write(summary)

            return True

        #  SHAP not available then fallback to coefficients
      
        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_).ravel()
            features = X.columns
        else:
            print("No coefficients found — cannot explain model.")
            return False

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        # Plot fallback
        plt.figure(figsize=(8, 5))
        plt.barh(imp_df["Feature"], imp_df["Importance"], color="#4D79FF", edgecolor="black")
        plt.gca().invert_yaxis()
        plt.title(f"Feature Importance — {model_name}")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{model_name}_simple.png", dpi=300)
        plt.close()

        return True

    except Exception as e:
        print(f"Explainability failed for {model_name}: {e}")
        return False

# Trending Products Visualization

def plot_trending_products(products, save_path="static/trending_products.png"):
    if not products:
        return

    top = products[:10]
    plt.figure(figsize=(10, 5))


    bars = plt.barh(top, range(len(top), 0, -1), color="#FF9933", edgecolor="black")

    #  Add labels on bars for clarity
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"Rank {int(len(top) - bar.get_width() + 1)}",
                 va="center", ha="left", fontsize=9, color="black")

    plt.title("Top Trending Products (Highest Selling First)", fontsize=13, fontweight="bold")
    plt.xlabel("Ranking Position")
    plt.ylabel("Product Names")


    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
