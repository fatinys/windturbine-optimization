from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the model

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get values from the form
        year_built = int(request.form['year_built'])
        hub_height = float(request.form['hub_height'])
        rotor_diameter = float(request.form['rotor_diameter'])
        location = request.form['location']

        # Create a DataFrame with the input
        input_data = pd.DataFrame({
            'Year Built': [year_built],
            'Hub Height': [hub_height],
            'Rotor Diameter': [rotor_diameter],
            'Location': [location]
        })

    
        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000,debug=True)