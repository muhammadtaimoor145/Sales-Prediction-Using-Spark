from flask import Flask, jsonify, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)
model = pickle.load(open('Decision_Tree.pkl', 'rb'))
transformation=pickle.load(open('Scaling.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    Item_Visibility=float(request.form['Item_Visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,Item_Visibility,item_type,item_mrp,outlet_type ]])
    trans=transformation.transform(X)
    Y_pred=model.predict(trans,round(2))
    
    return jsonify({'Prediction': float(Y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)