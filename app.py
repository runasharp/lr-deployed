from flask import Flask, request, render_template
import pickle
import model as md
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getresults', methods=['POST', 'GET'])
def get_delay():
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

        meanabs = round(md.sm.mean_absolute_error(md.Y_test, md.Y_pred), 2)
        medabs = round(md.sm.median_absolute_error(md.Y_test, md.Y_pred), 2)
        r2 = round(md.sm.r2_score(md.Y_test, md.Y_pred), 2)

        return render_template('result.html', prediction=prediction, meanabs=meanabs, medabs=medabs, r2=r2)


if __name__ == '__main__':
    # exec(open('model.py').read())
    app.debug = True
    app.run()
