from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__, template_folder=os.path.abspath('templates'))

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            def safe_int(value, default=0):
                try:
                    return int(value) if value is not None else default
                except ValueError:
                    return default

            data=CustomData(
                
                status=safe_int(request.form.get('Status')),
                duration=safe_int(request.form.get('Duration')),
                credit_history=safe_int(request.form.get('Credit History')),
                purpose=safe_int(request.form.get('Purpose')),
                amount=safe_int(request.form.get('Amount')),
                savings=safe_int(request.form.get('Savings')),
                employment_duration=safe_int(request.form.get('Employment Duration')),
                installment_rate=safe_int(request.form.get('Installment Rate')),
                personal_status_sex=safe_int(request.form.get('Personal Status Sex')),
                other_debtors=safe_int(request.form.get('Other Debtors')),
                present_residence=safe_int(request.form.get('Present Residence')),
                property=safe_int(request.form.get('Property')),
                age=safe_int(request.form.get('age')),
                other_installment_plans=safe_int(request.form.get('Other Installment Plans')),
                housing= safe_int(request.form.get('housing')),
                number_credits=safe_int(request.form.get('Number credits')),
                job=safe_int(request.form.get('job')),
                people_liable=safe_int(request.form.get('People Liable')),
                telephone=safe_int(request.form.get('Telephone')),
                foreign_worker=safe_int(request.form.get('Foreign Worker'))
            )
            pred_df=data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results=predict_pipeline.predict(pred_df)
            result = "Safe" if results[0] == 1 else "Not Safe"
            print("after Prediction")
            return render_template('home.html',results=result)
        except Exception as e:
            print("Error occurred:", str(e))
            return render_template('home.html', error=f"An error occurred during prediction: {str(e)}")


if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Template folder:", os.path.join(os.getcwd(), 'templates'))
    app.run(host='0.0.0.0', port=5001, debug=True)