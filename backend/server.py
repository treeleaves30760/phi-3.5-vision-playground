from flask import Flask, request, jsonify
from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

app = Flask(__name__)

model_id = "microsoft/Phi-3.5-vision-instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    load_in_8bit=True,
    _attn_implementation='eager'
)

# Load processor
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=1
)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    if not data or 'prompt' not in data or 'image_urls' not in data:
        return jsonify({"error": "Missing prompt or image URLs"}), 400

    prompt = data['prompt']
    image_urls = data['image_urls']

    images = []
    placeholder = ""

    for i, url in enumerate(image_urls, start=1):
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)
            placeholder += f"<|image_{i}|>\n"
        except Exception as e:
            return jsonify({"error": f"Failed to load image from URL {url}: {str(e)}"}), 400

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