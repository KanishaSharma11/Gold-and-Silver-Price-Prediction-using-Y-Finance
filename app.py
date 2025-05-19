from flask import Flask, render_template, request
from goldSilverPredictionUsingYFinance import load_data, preprocess_data, build_model, forecast_future
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    future_prediction = None
    trend_message = None
    selected_metal = None
    day_selected = None

    if request.method == 'POST':
        metal = request.form['metal']
        day_selected = request.form.get('day', '')
        day_selected = int(day_selected) if day_selected else None

        ticker = "GC=F" if metal == "Gold" else "SI=F"
        selected_metal = metal

        # Load and preprocess data
        df = load_data(ticker)
        X, y, scaler = preprocess_data(df)

        # Build and train model
        model = build_model(X)
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)

        # Today's prediction
        last_sequence = X[-1]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        prediction = model.predict(last_sequence)
        prediction = scaler.inverse_transform(prediction)[0][0]

        # Forecast next 30 days
        future_prices = forecast_future(model, df, scaler, days=30)

        # If user selected a specific day
        if day_selected and 1 <= day_selected <= 30:
            future_prediction = float(future_prices[day_selected - 1])

        # Determine trend (increase or decrease)
        if future_prices[-1] > prediction:
            trend_message = f"{metal} price is expected to increase over the next 30 days."
        else:
            trend_message = f"{metal} price is expected to decrease over the next 30 days."

    return render_template(
        'index.html',
        prediction=prediction,
        metal=selected_metal,
        future_prediction=future_prediction,
        trend_message=trend_message,
        day_selected=day_selected
    )

if __name__ == '__main__':
    app.run(debug=True)
