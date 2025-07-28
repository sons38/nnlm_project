from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from src.inference import NNLMInfer

app = Flask(__name__)
run_with_ngrok(app)
infer = NNLMInfer()

@app.route("/chat", methods=["POST"])
def chat():
    h = request.json.get("history",[])
    if len(h)<infer.cs: return jsonify({"error":"need length"}),400
    return jsonify({"response":infer.predict(h[-infer.cs:])})

if __name__=="__main__":
    app.run()
