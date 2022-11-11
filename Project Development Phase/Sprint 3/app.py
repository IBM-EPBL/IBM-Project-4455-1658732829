from flask import Flask, request, render_template
import flask
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

# import pickle files of models
knn = pickle.load(
    open('E:\Sem7\ibm\Sprint3\smart_lender\pickles\knn.pkl', 'rb'))
dtree = pickle.load(
    open('E:\Sem7\ibm\Sprint3\smart_lender\pickles\decisiontree.pkl', 'rb'))
nb = pickle.load(
    open(r'E:\Sem7\ibm\Sprint3\smart_lender\pickles\naivebayes.pkl', 'rb'))


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # inputParameters = request.form.to_dict()
    # input from form
    inputParameters = [int(x) for x in request.form.values()]
    # print("Input parameters")
    # print(inputParameters)  # type - list

    # append the list to the dataset
    df = pd.read_csv("main.csv")
    # print("df")
    # print(df.columns)

    df_new = df.append(pd.DataFrame(
        [inputParameters], columns=df.columns), ignore_index=True)
    # print("df_new")
    # print(df_new)
    
    # print(df_new.columns)
    scale = MinMaxScaler()
    scaled_data = pd.DataFrame(
        scale.fit_transform(df_new), columns=df_new.columns)
    # print("Type of scaled_data = ")
    # print(type(scaled_data))
    # print("Scaled_data")
    # print(scaled_data)

    scaled_inputs = scaled_data.iloc[-1]
    # print(type(scaled_inputs))

    # convert to numpy array
    np_array = scaled_inputs.to_numpy()
    # print(np_array)

    knn_prediction = knn.predict(np_array.reshape(1, -1))
    # print(knn_prediction[0])
    dtree_prediction = dtree.predict(np_array.reshape(1, -1))
    # print(dtree_prediction[0])
    naive_bayes_prediction = nb.predict(np_array.reshape(1, -1))
    # print(naive_bayes_prediction[0])

    op_knn = round(knn_prediction[0], 2)
    op_dtree = round(dtree_prediction[0], 2)
    op_naive_bayes = round(naive_bayes_prediction[0], 2)
    # print("---")
    # print(op_knn)
    # print(op_dtree)
    # print(op_naive_bayes)
    # print("---")
    
    zeros = 0
    ones = 0
    if (op_knn == 1):
        # print("1a")
        ones += 1
    else:
        # print("1b")
        zeros += 1

    if (op_dtree == 1):
        # print("2a")
        ones += 1
    else:
        # print("2b")
        zeros += 1

    if (op_naive_bayes == 1):
        # print("3a")
        ones += 1
    else:
        # print("3b")
        zeros += 1

    # print("ones=")
    # print(ones)
    # print("zeros=")
    # print(zeros)
    if (ones > zeros):
        output = "Your loan will get approved"
    else:
        output = "Your loan will not get approved"
    # print("output = " + output)
    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
