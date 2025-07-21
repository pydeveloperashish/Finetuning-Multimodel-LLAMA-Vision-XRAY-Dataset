from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import os

# Try to load the dataset - replace this with the actual dataset name if known
try:
    # First try the standard dataset loading
    dataset = load_dataset("unsloth/Radiology_mini", split = "train")
    print(f"Loaded dataset: {dataset}")
except Exception as e:
    print(f"Standard dataset loading failed: {e}")
  
# If we successfully loaded the dataset, display the first image
if 'dataset' in locals():
    try:
        # Try to access the image
        image = dataset[0]["image"]
        print("Successfully accessed image from dataset[0]")
        
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Display grayscale image on the left
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Dataset[0] Image (Grayscale)")
        axes[0].axis('off')
        
        # Display original colormap image on the right
        axes[1].imshow(image)
        axes[1].set_title("Dataset[0] Image (Original Colormap)")
        axes[1].axis('off')
        
        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
        
        # Show dataset caption if available
        if "caption" in dataset[0]:
            print(f"Image caption: {dataset[0]['caption']}")
            
    except Exception as e:
        print(f"Error accessing or displaying image: {e}")
    