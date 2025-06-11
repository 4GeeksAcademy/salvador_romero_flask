from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('../models/modelo_personalidad.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = [
        float(request.form['Time_spent_Alone']),
        float(request.form['Stage_fear']),
        float(request.form['Social_event_attendance']),
        float(request.form['Going_outside']),
        float(request.form['Drained_after_socializing']),
        float(request.form['Friends_circle_size']),
        float(request.form['Post_frequency'])
    ]
    prediccion = model.predict([datos])[0]

    return render_template('index.html', prediccion=f'Tu personalidad es: {prediccion}')

if __name__ == '__main__':
    app.run(debug=True)