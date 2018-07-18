from flask import Flask, request, jsonify
app = Flask(__name__)

prs = {}

@app.route('/')
def hello_world():
    return 'Hello Bhavana!'

@app.route('/github', methods=['POST'])
def github():
    print(request.data)
    return '', 200

@app.route('/status')
def status():
    return jsonify(prs), 200

@app.route('/<pr_number>/retest')
def status(pr_number):
    if prs[pr_number]
    return '', 200

if __name__ == '__main__':
    app.run()
