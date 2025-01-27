import os
from PIL import Image
from pathlib import Path

def make_gifs_loop(input_folder="q7", output_folder="q7_out", filter_pattern=None):
    """
    Convert GIFs in the input folder to continuously looping GIFs in the output folder.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    successful = 0
    failed = 0
    
    try:
        files = os.listdir(input_folder)
        print(f"Found {len(files)} files in {input_folder}")
        
        for filename in files:
            if filter_pattern:
                should_process = filter_pattern in filename
            else:
                should_process = filename.lower().endswith('.gif')
            
            if should_process:
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                try:
                    print(f"Processing: {filename}")
                    with Image.open(input_path) as img:
                        try:
                            img.seek(1)
                            is_animated = True
                        except EOFError:
                            is_animated = False
                        img.seek(0)
                        
                        if is_animated:
                            img.save(
                                output_path,
                                save_all=True,
                                loop=0,
                                optimize=True
                            )
                        else:
                            img.save(output_path)
                            
                    successful += 1
                    print(f"Successfully converted: {filename}")
                    
                except Exception as e:
                    failed += 1
                    print(f"Error processing {filename}: {str(e)}")
                    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return successful, failed
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful} files")
    print(f"Failed conversions: {failed} files")
    
    return successful, failed

if __name__ == "__main__":
    make_gifs_loop()