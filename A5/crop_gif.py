import os
from PIL import Image, ImageSequence
import argparse
from pathlib import Path

def crop_gif(input_path, output_path, top_crop):
    """
    Loads a GIF file, crops the top N pixels from all frames,
    and saves the result as a new GIF.
    
    Parameters:
    - input_path: Path to the input GIF file
    - output_path: Path to save the cropped GIF
    - top_crop: Number of pixels to crop from the top
    """
    try:
        # Open the GIF file
        img = Image.open(input_path)
        
        # Get dimensions
        width, height = img.size
        
        # Check if we have enough height to crop
        if height <= top_crop:
            print(f"Warning: {input_path} height ({height}) is less than or equal to crop amount ({top_crop}), skipping.")
            return False
        
        # New dimensions
        new_height = height - top_crop
        
        # Check if image is animated
        is_animated = hasattr(img, 'n_frames') and img.n_frames > 1
        
        if is_animated:
            frames = []
            durations = []
            
            # Process each frame
            for frame in ImageSequence.Iterator(img):
                # Crop the frame
                cropped_frame = frame.crop((0, top_crop, width, height))
                frames.append(cropped_frame)
                
                # Get duration for this frame
                try:
                    durations.append(img.info.get('duration', 100))
                except:
                    durations.append(100)  # Default 100ms
            
            # Save the animated GIF
            frames[0].save(
                output_path,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=durations,
                disposal=2,  # Clear previous frame
                loop=img.info.get('loop', 0)  # Preserve original loop setting
            )
        else:
            # Crop the single frame
            cropped_img = img.crop((0, top_crop, width, height))
            cropped_img.save(output_path, 'GIF')
        
        print(f"Cropped GIF saved to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_directory(input_dir, output_dir, top_crop):
    """
    Recursively finds all GIF files in input_dir and its subdirectories,
    crops them, and saves them to output_dir maintaining the directory structure.
    
    Parameters:
    - input_dir: Directory to search for GIFs
    - output_dir: Directory to save cropped GIFs
    - top_crop: Number of pixels to crop from the top
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for processed files
    success_count = 0
    fail_count = 0
    
    # Walk through the directory tree
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Check if file is a GIF
            if file.lower().endswith('.gif'):
                input_path = os.path.join(root, file)
                
                # Create corresponding output path with directory structure
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                output_path = os.path.join(output_subdir, file)
                
                # Process the GIF
                if crop_gif(input_path, output_path, top_crop):
                    success_count += 1
                else:
                    fail_count += 1
    
    print(f"\nProcessing complete!\nSuccessfully cropped: {success_count} GIFs\nFailed: {fail_count} GIFs")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Crop the top N pixels from all GIFs in a directory')
    parser.add_argument('input_dir', help='Input directory containing GIFs')
    parser.add_argument('output_dir', help='Output directory for cropped GIFs')
    parser.add_argument('top_crop', type=int, help='Number of pixels to crop from the top')
    
    args = parser.parse_args()
    
    # Process the directory
    process_directory(args.input_dir, args.output_dir, args.top_crop)