from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the model

@app.route('/', methods=['GET', 'POST'])
def home():
    return "Hallo"

if __name__ == '__main__':
    app.run(port=3000,debug=True)