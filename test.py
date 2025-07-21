import os
from unsloth import FastVisionModel
import torch
from transformers import TextStreamer
from PIL import Image

def main():
    # Load the LoRA model from the local directory
    model_path =  "devashish07/Multiodel-Radiology-Finetuing-LLAMA-3.2-11B"
    
    # Load the model and tokenizer
    print("Loading model and tokenizer...")
    
    # Explicitly set device_map to 'cuda' if available, otherwise 'cpu'
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",  # Base model
        adapter_name=model_path,                  # LoRA adapter
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map=device_map # Use the determined device
    )

    # Set model for inference
    FastVisionModel.for_inference(model)
    
    # Load your image (replace 'your_image.jpg' with the path to your image)
    image_path = "/sample-images/image.jpg"   # Update this to your image path
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with your actual image path.")
        return
    
    image = Image.open(image_path)
    
    # Create the instruction for the model
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."
    
    # Format the messages with the image and instruction
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    # Apply chat template and prepare inputs
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    # Get device
    print(f"Using device: {device_map}")
    
    # Tokenize inputs
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device_map)
    
    # Generate response
    print("\nGenerating response:\n")
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=128,
        use_cache=True, 
        temperature=1, 
        min_p=0.1
    )

if __name__ == "__main__":
    main() 