import numpy as np
import argparse
import torch
import os
from models import cls_model
from data_loader import get_data_loader
from utils import create_dir, viz_cls
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='cls', help='model to use')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=5, help="number of samples to visualize")
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    # Robustness analysis parameters
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to use for evaluation')
    parser.add_argument('--rotation', type=float, default=0, help='Rotation angle in degrees (around z-axis)')
    parser.add_argument('--rotation_axis', type=str, default='z', choices=['x', 'y', 'z'], help='Axis for rotation')
    parser.add_argument('--rotation_range', action='store_true', help='Test accuracy across a range of rotation angles')
    parser.add_argument('--point_range', action='store_true', help='Test accuracy across a range of point counts')
    parser.add_argument('--output_dir', type=str, default='./robustness_results', help='Directory to save results')
    
    # Dataloader parameters
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    
    return parser


def evaluate_model(model, test_dataloader, device, point_indices=None, rotation_angle=0, rotation_axis='z', class_names=None):
    """
    Evaluates the model on test data with optional point subsampling and rotation.
    
    Args:
        model: The model to evaluate
        test_dataloader: DataLoader with test data
        device: Device to use for tensors
        point_indices: Indices of points to use (if None, use all points)
        rotation_angle: Angle in degrees to rotate points
        rotation_axis: Axis for rotation ('x', 'y', or 'z')
        class_names: List of class names for visualization
        
    Returns:
        accuracy: Test accuracy
        viz_data: Visualization data for successful and failed classifications
    """
    correct_obj = 0
    num_obj = 0
    
    # For visualization
    viz_data = {
        'successes': {i: [] for i in range(len(class_names))} if class_names else None,
        'failures': {i: [] for i in range(len(class_names))} if class_names else None
    }
    
    # Create rotation matrix if needed
    if rotation_angle != 0:
        rot = R.from_euler(rotation_axis, rotation_angle, degrees=True)
        rot_matrix = torch.tensor(rot.as_matrix(), dtype=torch.float32, device=device)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(device)
            labels = labels.to(device).to(torch.long)
            
            # Apply point subsampling if specified
            if point_indices is not None:
                point_clouds = point_clouds[:, point_indices, :]
            
            # Apply rotation if specified
            if rotation_angle != 0:
                # Rotate each point cloud
                batch_size, num_points, _ = point_clouds.shape
                point_clouds = torch.bmm(point_clouds, rot_matrix.expand(batch_size, 3, 3))
            
            # Make predictions
            predictions = model(point_clouds)
            pred_labels = torch.argmax(predictions, dim=1)
            
            # Collect visualization data if class_names is provided
            if class_names and viz_data:
                for i in range(len(labels)):
                    is_correct = pred_labels[i].item() == labels[i].item()
                    true_label = labels[i].item()
                    pred_label = pred_labels[i].item()
                    
                    sample_data = {
                        'points': point_clouds[i].cpu().numpy(),
                        'true_label': true_label,
                        'pred_label': pred_label
                    }
                    
                    if is_correct and len(viz_data['successes'][true_label]) < 5:
                        viz_data['successes'][true_label].append(sample_data)
                    elif not is_correct and len(viz_data['failures'][true_label]) < 5:
                        viz_data['failures'][true_label].append(sample_data)
            
            # Update metrics
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]
    
    # Calculate accuracy
    accuracy = correct_obj / num_obj
    
    return accuracy, viz_data


def test_rotation_range(model, test_dataloader, device, axis='z', output_dir='./robustness_results'):
    """Tests the model across a range of rotation angles"""
    angles = np.arange(0, 360, 15)  # Test rotations every 15 degrees
    accuracies = []
    
    print(f"Testing rotation robustness around {axis}-axis...")
    for angle in angles:
        acc, _ = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            device=device,
            rotation_angle=angle,
            rotation_axis=axis
        )
        accuracies.append(acc * 100)  # Convert to percentage
        print(f"  Rotation: {angle}° - Accuracy: {acc:.4f}")
    
    # Create directory for results
    create_dir(output_dir)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(angles, accuracies, marker='o')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Accuracy vs. Rotation Angle ({axis}-axis)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/rotation_robustness_{axis}axis.png")
    plt.close()
    
    # Save results as CSV
    results = np.column_stack((angles, accuracies))
    np.savetxt(f"{output_dir}/rotation_robustness_{axis}axis.csv", results, delimiter=',', 
               header='angle,accuracy', comments='')
    
    return angles, accuracies


def test_point_sampling_range(model, test_dataloader, device, output_dir='./robustness_results'):
    """Tests the model across a range of point counts"""
    # Test with different numbers of points
    point_counts = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    accuracies = []
    
    print("Testing point count robustness...")
    for num_points in point_counts:
        # Randomly sample points
        indices = np.random.choice(10000, num_points, replace=False)
        
        acc, _ = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            device=device,
            point_indices=indices
        )
        accuracies.append(acc * 100)  # Convert to percentage
        print(f"  Points: {num_points} - Accuracy: {acc:.4f}")
    
    # Create directory for results
    create_dir(output_dir)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(point_counts, accuracies, marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy vs. Number of Points')
    plt.grid(True)
    plt.savefig(f"{output_dir}/point_count_robustness.png")
    plt.close()
    
    # Save results as CSV
    results = np.column_stack((point_counts, accuracies))
    np.savetxt(f"{output_dir}/point_count_robustness.csv", results, delimiter=',', 
               header='num_points,accuracy', comments='')
    
    return point_counts, accuracies


def main(args):
    """
    Loads trained model and evaluates on test set with specified robustness tests.
    """
    # Create model directory
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir + "/" + args.task
    
    # Create output directory
    create_dir(args.output_dir)

    # Initialize model
    model = cls_model().to(args.device)
    
    # Load model
    model_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    print("Successfully loaded checkpoint from {}".format(model_path))
    
    # Test model
    test_dataloader = get_data_loader(args=args, train=False)
    
    # Class names
    class_names = ['chair', 'vase', 'lamp']
    
    # Run robustness experiments based on args
    if args.rotation_range:
        # Test rotation robustness
        angles, accuracies = test_rotation_range(
            model=model,
            test_dataloader=test_dataloader,
            device=args.device,
            axis=args.rotation_axis,
            output_dir=args.output_dir
        )
        print(f"Rotation robustness results saved to {args.output_dir}")
        
    elif args.point_range:
        # Test point count robustness
        point_counts, accuracies = test_point_sampling_range(
            model=model,
            test_dataloader=test_dataloader,
            device=args.device,
            output_dir=args.output_dir
        )
        print(f"Point count robustness results saved to {args.output_dir}")
        
    else:
        # Standard evaluation with optional point sampling and rotation
        # Sample points if num_points is less than 10000
        point_indices = None
        if args.num_points < 10000:
            point_indices = np.random.choice(10000, args.num_points, replace=False)
        
        # Evaluate model
        accuracy, viz_data = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            device=args.device,
            point_indices=point_indices,
            rotation_angle=args.rotation,
            rotation_axis=args.rotation_axis,
            class_names=class_names
        )
        
        print(f"Test accuracy with {args.num_points} points and {args.rotation}° {args.rotation_axis}-axis rotation: {accuracy:.4f}")
        
        # Create visualization directory
        vis_dir = f"{args.output_dir}/np{args.num_points}_rot{args.rotation}{args.rotation_axis}"
        create_dir(vis_dir)
        
        # Visualize successful classifications
        print("Visualizing successful classifications...")
        for class_idx, class_name in enumerate(class_names):
            sample_data = viz_data['successes'][class_idx]
            for i, data in enumerate(sample_data):
                outfile = f'{vis_dir}/success_{class_name}_{i}.gif'
                viz_cls(data['points'], data['true_label'], data['pred_label'], class_names, outfile, args.device)

        # Visualize failed classifications
        print("Visualizing failed classifications...")
        for class_idx, class_name in enumerate(class_names):
            sample_data = viz_data['failures'][class_idx]
            for i, data in enumerate(sample_data):
                outfile = f'{vis_dir}/failure_{class_name}_{i}.gif'
                viz_cls(data['points'], data['true_label'], data['pred_label'], class_names, outfile, args.device)
        
        print("Visualization complete.")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)