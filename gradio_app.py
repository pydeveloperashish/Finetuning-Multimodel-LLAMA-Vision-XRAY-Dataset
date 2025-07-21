import os
import gradio as gr
from unsloth import FastVisionModel
import torch
from transformers import TextStreamer
from PIL import Image

# Load model and tokenizer once
model_path = "./lora_model"
device_map = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[LOG] Loading model on device: {device_map}")

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    adapter_name=model_path,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    device_map=device_map
)

FastVisionModel.for_inference(model)
print("[LOG] Model and tokenizer loaded.")

instruction = "You are an expert radiographer. Describe accurately what you see in this image."

def analyze_image(image):
    if image is None:
        return "No image selected!"
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device_map)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=1,
            min_p=0.1
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if instruction in output_text:
            output_text = output_text.split(instruction, 1)[1]
        
        prefixes_to_remove = [
            "You are an expert radiographer. Describe accurately what you see in this image.",
            "Describe accurately what you see in this image.",
            "assistant",
            "Assistant:",
            "I am an expert radiographer."
        ]
        
        for prefix in prefixes_to_remove:
            if output_text.startswith(prefix):
                output_text = output_text[len(prefix):]
        
        output_text = output_text.lstrip(":., \n")
        
        return output_text.strip()
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Radiology Image Analyzer\nUpload an image to get an AI analysis.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            output = gr.Textbox(label="Model Output", lines=10)
    
    analyze_btn = gr.Button("Analyze", variant="primary")
    analyze_btn.click(analyze_image, inputs=image_input, outputs=output)

demo.launch(share=True) 