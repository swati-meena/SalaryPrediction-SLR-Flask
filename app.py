from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


with open('linear_regression_model.pkl','rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        years_exp = float(request.form['experience'])
        prediction = model.predict(np.array([[years_exp]]))
        predicted_salary = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f'Predicted Salary: ₹{predicted_salary}')
   
    except:
        return render_template('index.html',prediction_text="Please enter a valid number.")
        
if __name__ == '__main__':
    app.run(debug=True)