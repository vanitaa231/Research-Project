
import pandas as pd
from dataLoader import load_data
from forecasting import forecast_trending_products


def recommend_products(Customer_Name, top_n=10, explain=True):

    # Load data
    _, df = load_data()

    # Validate required columns
    if not all(col in df.columns for col in ["Customer_Name", "Product"]):
        return [], ["Dataset missing required columns: Customer_Name / Product"]

    # Customer purchase history
    customer_df = df[df["Customer_Name"] == Customer_Name]

    if customer_df.empty:
        return [], [f"No purchase history found for customer: {Customer_Name}"]

    purchased_products = set(customer_df["Product"].unique())

    # All other products
    all_products = set(df["Product"].unique())
    candidate_products = list(all_products - purchased_products)

    if not candidate_products:
        return [], ["Customer has already purchased all products!"]

 
    # Find similar customers

    similar_customers = df[df["Product"].isin(purchased_products)]["Customer_Name"].unique()

    # Co-purchase scores
   
    co_purchase_score = {}

    for p in candidate_products:
        count = df[(df["Customer_Name"].isin(similar_customers)) &
                   (df["Product"] == p)].shape[0]
        co_purchase_score[p] = count

    # Normalize
    max_cp = max(co_purchase_score.values()) or 1
    for p in co_purchase_score:
        co_purchase_score[p] /= max_cp

    # Trending scores (Taken from forecasting)
   
    trending_list = forecast_trending_products(top_n=50)
    trending_score = {p: (1.0 / (i + 1)) for i, p in enumerate(trending_list)}

    max_ts = max(trending_score.values()) if trending_score else 1
    for p in trending_score:
        trending_score[p] /= max_ts

    # Final scores

    final_scores = {}

    for p in candidate_products:
        final_scores[p] = (
            0.6 * co_purchase_score.get(p, 0) +
            0.4 * trending_score.get(p, 0)
        )

    sorted_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_products = [p for p, s in sorted_products][:top_n]

    #  Explaination
 
    explanations = []
    if explain:
        for p in recommended_products:
            parts = []

            if co_purchase_score.get(p, 0) > 0:
                parts.append("customers with similar buying patterns purchased this")

            if trending_score.get(p, 0) > 0:
                parts.append("this product is trending for future demand")

            if parts:
                explanations.append(f"{p}: Recommended because " + " and ".join(parts) + ".")
            else:
                explanations.append(f"{p}: Recommended based on overall popularity.")

    return recommended_products, explanations


def recommend_products_simple(top_n=5):
    return forecast_trending_products(top_n=top_n)
