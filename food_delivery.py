import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

app = Flask(__name__)

data = pd.read_csv("d:/Dev/Machine Learning/Multiple Regression/food_delivery.csv")
features = data[["Distance_KM","Traffic_Level_Score","Prep_Time_Mins"]]
target = data[["Delivery_Time_Mins"]]

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state= 42)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print("________________________________________")
print("|         Model Test Results           |")
print("|______________________________________|")
print(f"Average error: {round(mae,1)} minutes")
print(f"Accuracy: {r2:.2f}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_dist = float(request.form["dist"])
    user_traffic = float(request.form["traffic"])
    user_prep = float(request.form["time"])
    
    user_df = pd.DataFrame(
        [[user_dist, user_traffic, user_prep]], 
        columns=["Distance_KM", "Traffic_Level_Score", "Prep_Time_Mins"]
    )
    
    prediction = model.predict(user_df)
    raw_time = float(prediction[0][0])
    
    hours = int(raw_time // 60)
    minutes = round(raw_time % 60)
    
    output_text = f"Your order will arrive in {hours} hours and {minutes} minutes"
    
    return render_template("index.html", prediction_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)
