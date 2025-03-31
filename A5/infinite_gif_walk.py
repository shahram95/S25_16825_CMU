import os
from PIL import Image
from pathlib import Path

def make_gifs_loop(input_folder="robustness_results_seg/", output_folder="robustness_results_seg_loop/", filter_pattern=None):
    """
    Recursively convert GIFs in the input folder and its subdirectories to continuously 
    looping GIFs, maintaining the same directory structure in the output folder.
    """
    # Create the base output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    successful = 0
    failed = 0
    
    try:
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(input_folder):
            # Calculate the relative path from the input folder
            rel_path = os.path.relpath(root, input_folder)
            
            # Create the corresponding output directory structure
            if rel_path != '.':
                output_dir = os.path.join(output_folder, rel_path)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            else:
                output_dir = output_folder
            
            # Process files in the current directory
            gif_files = [f for f in files if f.lower().endswith('.gif')]
            if filter_pattern:
                gif_files = [f for f in gif_files if filter_pattern in f]
            
            print(f"Found {len(gif_files)} GIF files in {root}")
            
            for filename in gif_files:
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_dir, filename)
                
                try:
                    print(f"Processing: {input_path}")
                    with Image.open(input_path) as img:
                        try:
                            img.seek(1)  # Check if the GIF has more than one frame
                            is_animated = True
                        except EOFError:
                            is_animated = False
                        img.seek(0)  # Go back to the first frame
                        
                        if is_animated:
                            img.save(
                                output_path,
                                save_all=True,
                                loop=0,  # 0 means infinite loop
                                optimize=True
                            )
                        else:
                            img.save(output_path)
                            
                    successful += 1
                    print(f"Successfully converted: {filename}")
                    
                except Exception as e:
                    failed += 1
                    print(f"Error processing {input_path}: {str(e)}")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return successful, failed
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions: {failed} files")
    
    return successful, failed

if __name__ == "__main__":
    # You can specify your input and output directories here
    make_gifs_loop()