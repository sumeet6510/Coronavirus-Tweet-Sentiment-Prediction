import pickle
from flask import Flask, render_template, request

model_load = pickle.load(open('Sentiment Analysis', 'rb'))

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def submit():
    if request.method == 'POST':
        msg = request.form['Tweet']
        msg_lst = []
        msg_lst.append(msg)
        result = model_load.predict(msg_lst)
        return render_template('index.html', final_result = result)

if __name__ == '__main__':
    app.run()
