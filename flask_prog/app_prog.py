from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Error: Division by zero"
    return x / y

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/perform_action', methods=['POST'])
def perform_action():
    data = request.get_json()
    action = data.get('action')
    x = data.get('x')
    y = data.get('y')

    if x is None or y is None:
        return jsonify({'error': 'Please provide values for x and y'}), 400

    try:
        x = float(x)
        y = float(y)
    except ValueError:
        return jsonify({'error': 'Invalid value provided for x or y. Please provide numeric values.'}), 400

    if action == "QA 1":
        result = add(x, y)
    elif action == "QA 2":
        result = multiply(x, y)
    elif action == "QA 3":
        result = divide(x, y)
    else:
        result = f"Unknown action: {action}"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

