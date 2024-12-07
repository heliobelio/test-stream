
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Hugging Face's DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Tokenize input and generate response
    inputs = tokenizer.encode(question, return_tensors="pt")
    reply_ids = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
