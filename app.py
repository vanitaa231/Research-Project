from flask import Flask, render_template, request
import os
from forecasting import forecast_sales, load_data, forecast_trending_products
from recommendation import recommend_products

app = Flask(__name__)


# Load Data Once

daily_sales, df = load_data()
metrics, future_df = forecast_sales(daily_sales)
trending_products = forecast_trending_products(top_n=10)


# Load Explanation TXT Files

def load_text(filename):
    path = os.path.join("static", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return "Explanation file not found."

lr_explanation = load_text("LinearRegression_explanation.txt")
trend_explanation = load_text("trending_explanation.txt")


# Dashboard Route

@app.route("/", methods=["GET", "POST"])
def dashboard():
    customer = None
    combined_recs = []

    if request.method == "POST":
        customer = request.form.get("customer")
        if customer:
            try:
                recs, explanations = recommend_products(customer, top_n=10, explain=True)
                combined_recs = list(zip(recs, explanations))
            except Exception as e:
                print("Recommendation Error:", e)
                combined_recs = []

    return render_template(
        "index.html",
        metrics=metrics,
        future=future_df,
        trending=trending_products,
        combined_recs=combined_recs,
        customer_name=customer, 
        lr_explanation=lr_explanation,
        trend_explanation=trend_explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
