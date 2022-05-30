from flask import Flask,render_template,request
from flask_cors import cross_origin
from Preprocessing import Preprocessor
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('rf_model_saved.sav', 'rb'))

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/test",methods=['POST'])
def testForecast():
    try:
        preprocessorobj = Preprocessor()
        test = preprocessorobj.readDataset(path='Testing_file/test_data.csv')
        test = preprocessorobj.creatingNewFeatures(test)
        test = test.drop(['Prediction'], axis=1)
 
        ## removing multicollinear features in test data
        test = test.drop(['dayofyear', 'weekofyear'], axis=1)

        prediction_result = preprocessorobj.predictingTestData(data=test,filename='test_data.csv')
        result = prediction_result

        date = result['Date']
        price = result['Prediction']
        results = []
        for i in range(len(date)):
            mydict = {'Date': date[i], 'Price': price[i]}
            results.append(mydict)

        return render_template('results.html', results=results)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))


@app.route("/single",methods=['POST'])
def singleForecast():
    file = request.files['Filename']
    print("************************************************************************")
    print(type(file))
    print(file)
    file.save(os.path.join(r'Testing_file',file.filename))
    try:
        preprocessorobj = Preprocessor()
        test = preprocessorobj.readDataset(path='Testing_file/{}'.format(file.filename))
        test = preprocessorobj.creatingNewFeatures(test)
        test = test.drop(['Prediction','dayofyear', 'weekofyear'], axis=1)

        #test = preprocessorobj.PreparingForPrediction(date)
        prediction_result = preprocessorobj.predictingTestData(data=test,filename=file.filename)
        result = prediction_result

        date = result['Date']
        price = result['Prediction']
        results = []
        for i in range(len(date)):
            mydict = {'Date': date[i], 'Price': price[i]}
            results.append(mydict)

        return render_template('results.html', results=results)
    except Exception as e:
        raise Exception(f"(app.py) - Something went wrong"+str(e))

if __name__ == "__main__":
    app.run(port=9000,debug=True)