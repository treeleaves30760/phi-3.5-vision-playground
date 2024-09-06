from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

app = Flask(__name__)

model_id = "microsoft/Phi-3.5-vision-instruct"

# Load model and processor
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=4
)

@app.route('/submit', methods=['POST'])
def submit():
    if 'prompt' not in request.form:
        return jsonify({"error": "No prompt provided"}), 400
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    prompt = request.form['prompt']
    files = request.files.getlist('files')

    images = []
    placeholder = ""

    for i, file in enumerate(files, start=1):
        image = Image.open(io.BytesIO(file.read()))
        images.append(image)
        placeholder += f"<|image_{i}|>\n"

    messages = [
        {"role": "user", "content": placeholder + prompt},
    ]

    full_prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(full_prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)