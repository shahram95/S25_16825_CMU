from PIL import Image, ImageSequence
import numpy as np

def change_colors(input_path, output_path, new_color=(255, 0, 0)):
    """
    Loads a GIF file, keeps white background, and changes all other pixels to new_color.
    Preserves animation if present.
    
    Parameters:
    - input_path: Path to the input GIF file
    - output_path: Path to save the modified GIF
    - new_color: Tuple of (R, G, B) values for the new color, default is red
    """
    # Open the GIF file
    img = Image.open(input_path)
    
    # Check if image is animated
    is_animated = hasattr(img, 'n_frames') and img.n_frames > 1
    
    if is_animated:
        # For animated GIFs, process each frame
        frames = []
        
        # Store the original duration between frames
        durations = []
        
        for frame in ImageSequence.Iterator(img):
            # Convert frame to RGBA
            frame = frame.convert('RGBA')
            
            # Convert to numpy array
            frame_data = np.array(frame)
            
            # Create a mask for non-white pixels
            tolerance = 5
            white_mask = (frame_data[:,:,0] > 255 - tolerance) & \
                         (frame_data[:,:,1] > 255 - tolerance) & \
                         (frame_data[:,:,2] > 255 - tolerance)
            
            # Get alpha channel
            alpha = frame_data[:,:,3]
            
            # Create new frame data
            new_frame_data = frame_data.copy()
            
            # Change colors of non-white pixels
            for i in range(3):
                new_frame_data[:,:,i] = np.where(white_mask, 255, new_color[i])
            
            # Preserve alpha channel
            new_frame_data[:,:,3] = alpha
            
            # Create new frame and append to frames list
            new_frame = Image.fromarray(new_frame_data)
            frames.append(new_frame)
            
            # Get duration for this frame
            try:
                durations.append(img.info.get('duration', 100))
            except:
                durations.append(100)  # Default 100ms if not specified
        
        # Save the animated GIF
        frames[0].save(
            output_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=durations,
            disposal=2,  # Clear previous frame
            loop=0  # Loop forever
        )
    else:
        # Process single frame image
        # Convert to RGBA
        img = img.convert('RGBA')
        
        # Convert to numpy array
        data = np.array(img)
        
        # Create a mask for non-white pixels
        tolerance = 5
        white_mask = (data[:,:,0] > 255 - tolerance) & \
                     (data[:,:,1] > 255 - tolerance) & \
                     (data[:,:,2] > 255 - tolerance)
        
        # Get alpha channel
        alpha = data[:,:,3]
        
        # Create new data
        new_data = data.copy()
        
        # Change colors of non-white pixels
        for i in range(3):
            new_data[:,:,i] = np.where(white_mask, 255, new_color[i])
        
        # Preserve alpha channel
        new_data[:,:,3] = alpha
        
        # Create and save new image
        new_img = Image.fromarray(new_data)
        new_img.save(output_path, 'GIF')
    
    print(f"Image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    name = "failure_vase_4"
    R = (255, 0, 0)
    G = (0, 255, 0)
    B = (0, 0, 255)
    input_gif = "./output/cls/{}.gif".format(name)
    output_gif = "./output/q4/pointnet++/{}.gif".format(name)
    output_gif2 = "./output/q4/dgcnn/{}.gif".format(name)
    output_gif3 = "./output/q4/pointtransformer/{}.gif".format(name)
    
    # Change all non-white pixels to blue (R, G, B)
    change_colors(input_gif, output_gif, new_color=G)
    change_colors(input_gif, output_gif2, new_color=G)
    change_colors(input_gif, output_gif3, new_color=G)