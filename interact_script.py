from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_id = "microsoft/Phi-3.5-vision-instruct"

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    _attn_implementation='eager'
)

processor = AutoProcessor.from_pretrained(model_id,
                                          trust_remote_code=True,
                                          num_crops=4
                                          )

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1, 20):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder+"Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}

with torch.inference_mode():
    generate_ids = model.generate(**inputs,
                                  eos_token_id=processor.tokenizer.eos_token_id,
                                  **generation_args
                                  )

# remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]

print(response)