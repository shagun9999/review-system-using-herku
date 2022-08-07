import numpy as np
from flask import Flask, request, jsonify, render_template
 


app = Flask(__name__)
model_rf = pickle.load(open('commentnlp.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Review = request.args.get('Review')
    input_data = [Review] 
    input_data = cv.transform(input_data).toarray()
    input_pred = model_rf.predict(input_data)
    input_pred = input_pred.astype(int)
    print(input_pred)


    if input_pred[0]==1:
      prediction_text="Review is Positive"
    else:
      prediction_text="Review is Negative"
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
