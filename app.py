from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("Students_Placed_Logistic_SC_new.pkl", "rb"))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        cgpa = float(request.form["cgpa"])
        iq = int(request.form["iq"])
        profile_score = int(request.form["profile_score"])

        # Make a prediction using the model
        prediction = model.predict(pd.DataFrame([[cgpa, iq, profile_score]], columns=['cgpa', 'iq', 'profile_score']))

        # Return the result
        if prediction[0] == 0:
            return render_template('index.html', prediction_text="Sorry to say, You have not been Placed")
        elif prediction[0] == 1:
            return render_template('index.html', prediction_text="Congratulations, You have been Placed")

    return render_template('index.html')

if __name__ == "__main__":
    # Run the Flask app with specified host and port
    app.run(host='127.0.0.1', port=5000, debug=True)
