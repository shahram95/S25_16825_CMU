from q1_1 import create_render

def main():
    color1 = [0.0, 1.0, 0.0]
    color2 = [1.0, 1.0, 1.0]
    create_render(model_path='data/cow.obj', render_size=1024, gradient_colors=(color1, color2), output_dir="output", output_name="q3.gif", camera_height=20.0, rotation_step=5, animation_fps=30)

if __name__ == "__main__":
    main()    