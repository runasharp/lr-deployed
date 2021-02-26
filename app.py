from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/getresults', methods=['POST', 'GET'])
def get_results():
    if request.method == 'POST':

        x1 = request.form['x1']
        x2 = request.form['x2']
        x3 = request.form['x3']
        x4 = request.form['x4']
        x5 = request.form['x5']

        inputs = [[x1, x2, x3, x4, x5]]

        pkl_file = open('regr.pkl', 'rb')
        regr = pickle.load(pkl_file)
        prediction = int(regr.predict(inputs))
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.debug = True
    app.run()
