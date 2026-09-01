import os
from PIL import Image

def combine_images_to_pdf(output_pdf_name="combined_output.pdf"):
    # Get the Present Working Directory (PWD)
    pwd = os.getcwd()
    print(f"Scanning directory: {pwd}")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    image_paths = []

    # 1. Explore all folders and subfolders
    for root, dirs, files in os.walk(pwd):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    # Sort paths alphabetically so images appear in order
    image_paths.sort()

    if not image_paths:
        print("No images found in the current directory or its subfolders.")
        return

    print(f"Found {len(image_paths)} images. Converting to PDF...")

    # 2. Open images and convert them to RGB mode
    loaded_images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            # Convert to RGB because PDFs cannot handle RGBA (transparency) properly
            if img.mode != 'RGB':
                img = img.convert('RGB')
            loaded_images.append(img)
        except Exception as e:
            print(f"Skipping broken image {path}: {e}")

    if not loaded_images:
        print("No valid images could be loaded.")
        return

    # 3. Save all images into a single PDF
    first_image = loaded_images[0]
    rest_of_images = loaded_images[1:]
    
    output_path = os.path.join(pwd, output_pdf_name)
    first_image.save(output_path, save_all=True, append_images=rest_of_images)
    
    print(f"Success! Created PDF at: {output_path}")

if __name__ == "__main__":
    combine_images_to_pdf()
