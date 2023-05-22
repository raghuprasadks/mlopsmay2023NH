from flask import Flask
app = Flask(__name__)

@app.route("/")
def welcome():
    return "<h1>Welcome to Flask</h1>"

@app.route("/hello")
def helloworld():
    return "<h1>Hello World</h1>"

if __name__ == "__main__":
	app.run(debug=True,host="0.0.0.0", port=5000)