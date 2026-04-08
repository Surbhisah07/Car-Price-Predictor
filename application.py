from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load dataset (for dropdown values)
car = pd.read_csv("Cleaned Car.csv")

# Load trained pipeline model
with open("LinearRegressionModel.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    company_models = {
        c: sorted(car[car['company'] == c]['name'].unique())
        for c in companies
    }

    prediction = None

    if request.method == "POST":
        company = request.form.get("company")
        car_model = request.form.get("car_model")
        year = int(request.form.get("year"))
        if year < 1990 or year > 2025:
            return "Invalid year"
        fuel_type = request.form.get("fuel_type")
        kms_driven = int(request.form.get("kms_driven"))

        input_df = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        prediction = model.predict(input_df)[0]
        prediction = f"{int(max(0, prediction)):,}"

    return render_template(
        "index.html",
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        company_models=company_models,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)