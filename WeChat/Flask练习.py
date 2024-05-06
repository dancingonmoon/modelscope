from flask import Flask
from markupsafe import escape

app = Flask(__name__)
@app.route("/")
def index():
    return 'Index Page'
@app.route("/hello")
def hello():
    return "<p>Hello , World!</p>"
@app.route("/<name>")
def hello_world(name):
    return f"<p>Hello {escape(name)} , World!</p>"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
    # app.run(host="0.0.0.0", port=8080, debug=True)