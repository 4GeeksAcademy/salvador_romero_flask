from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('modelo_personalidad.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    stage_fear = 1 if request.form['Stage_fear'] == 'Yes' else 0
    drained = 1 if request.form['Drained_after_socializing'] == 'Yes' else 0

    datos = [
        float(request.form['Time_spent_Alone']),
        stage_fear,
        float(request.form['Social_event_attendance']),
        float(request.form['Going_outside']),
        drained,
        float(request.form['Friends_circle_size']),
        float(request.form['Post_frequency'])
    ]
    prediccion = model.predict([datos])[0]

    personalidad = "Introvert" if prediccion == 1 else "Extrovert"

    return render_template('index.html', prediccion=f'Your personality is: {personalidad}')

if __name__ == '__main__':
    app.run(debug=True)