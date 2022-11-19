from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import pandas as pd
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
# API_KEY = "27e141c7-5cbf-46cd-9cc9-faff2e7ebc9e"


# maatthavaendiyadhu
API_KEY = "3Pexm294mcuQRhjEJIBP6NRXfKfO713c9XnZM6UP4XoH"

token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
                                                                                 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + mltoken}

# token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={
#                                "apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
# mltoken = token_response.json()["access_token"]

# header = {'Content-Type': 'application/json',
#           'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)

rgressor = pickle.load(open(
    r'E:\Sem7\ibm\Sprint3\smart_lender\pickles\randomforestregressor.pkl', 'rb'))


@app.route('/')
def start():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    if request.method == "POST":
        inputParameters = [int(x) for x in request.form.values()]
        print("Applicant Income {}".format(inputParameters[5]))
        print("Input parameters")
        print(inputParameters)  # type - list

        inputParameters[5] = float(inputParameters[5])
        inputParameters[6] = float(inputParameters[6])

        inputList = [inputParameters]

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
        payload_scoring = {"input_data": [{"fields": [["Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome",
                                                       "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"]], "values": inputList}]}
        # maatthavaendiyadhu
        response_scoring = requests.post('https://eu-gb.ml.cloud.ibm.com/ml/v4/deployments/e1d22aab-8a1c-4758-83f4-9e7ee678f21f/predictions?version=2022-11-19', json=payload_scoring,
                                         headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        print(response_scoring.json())
        prediction = response_scoring.json()["predictions"][0]["values"][0][0]
        print(prediction)

        loan = rgressor.predict(np.array(inputParameters[5]).reshape(1, -1))
        # op_rgr = round(loan[0], 2)
        op_rgr = int(loan[0])

        if (prediction == 'Y'):
            output = "Your loan will be approved"
        else:
            output = "For this amount, your loan might not be approved. \nIf you apply " + \
                str(op_rgr) + ", your loan will be approved."
        return render_template('index.html', prediction_text=output)

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
