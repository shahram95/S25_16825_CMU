import numpy as np
import argparse
import torch
import os
import matplotlib.pyplot as plt
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from scipy.spatial.transform import Rotation as R

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./robustness_results_seg')

    # Robustness analysis parameters
    parser.add_argument('--rotation', type=float, default=0, help='Rotation angle in degrees (around z-axis)')
    parser.add_argument('--rotation_axis', type=str, default='z', choices=['x', 'y', 'z'], help='Axis for rotation')
    parser.add_argument('--rotation_range', action='store_true', help='Test accuracy across a range of rotation angles')
    parser.add_argument('--point_range', action='store_true', help='Test accuracy across a range of point counts')
    
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')

    return parser


# def evaluate_model(model, test_data, test_label, device, batch_size=32, point_indices=None, rotation_angle=0, rotation_axis='z'):
#     """
#     Evaluates the segmentation model with specified modifications.
    
#     Args:
#         model: The segmentation model
#         test_data: Tensor of test data
#         test_label: Tensor of test labels
#         device: Device to use for evaluation
#         batch_size: Batch size for evaluation
#         point_indices: Indices of points to use (if None, use all points)
#         rotation_angle: Angle in degrees to rotate points
#         rotation_axis: Axis for rotation ('x', 'y', or 'z')
        
#     Returns:
#         test_accuracy: Overall test accuracy
#         object_accuracies: List of per-object accuracies
#         pred_label: Predicted labels
#     """
#     # Apply point subsampling if specified
#     if point_indices is not None:
#         test_data = test_data[:, point_indices, :]
#         test_label = test_label[:, point_indices]
    
#     # Apply rotation if specified
#     if rotation_angle != 0:
#         # Create rotation matrix
#         rot = R.from_euler(rotation_axis, rotation_angle, degrees=True)
#         rot_matrix = torch.tensor(rot.as_matrix(), dtype=torch.float32, device=device)
        
#         # Apply rotation to each object
#         batch_size = test_data.shape[0]
#         test_data = torch.bmm(
#             test_data.to(device),
#             rot_matrix.expand(batch_size, 3, 3)
#         ).cpu()
    
#     # Process data in batches to predict segmentation
#     num_samples = test_data.shape[0]
#     num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
#     pred_label = torch.zeros_like(test_label)
    
#     # Get model predictions
#     model.eval()
#     with torch.no_grad():
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, num_samples)
            
#             batch_data = test_data[start_idx:end_idx].to(device)
#             batch_pred = model(batch_data)
#             pred_label[start_idx:end_idx] = torch.argmax(batch_pred, dim=2).cpu()
    
#     # Calculate overall accuracy
#     test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    
#     # Calculate per-object accuracies
#     object_accuracies = []
#     for i in range(num_samples):
#         acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / test_label[i].size(0)
#         object_accuracies.append((i, acc))
    
#     return test_accuracy, object_accuracies, pred_label

def evaluate_model(model, test_data, test_label, device, batch_size=32, point_indices=None, rotation_angle=0, rotation_axis='z'):
    """
    Memory-efficient evaluation that processes smaller batches when rotation is applied.
    """
    # Apply point subsampling if specified
    if point_indices is not None:
        test_data = test_data[:, point_indices, :]
        test_label = test_label[:, point_indices]
    
    # Create rotation matrix if needed (on CPU)
    if rotation_angle != 0:
        rot = R.from_euler(rotation_axis, rotation_angle, degrees=True)
        rot_matrix = torch.tensor(rot.as_matrix(), dtype=torch.float32)
    
    # Process data in batches to predict segmentation
    num_samples = test_data.shape[0]
    
    # Use smaller batches for rotation to avoid memory issues
    if rotation_angle != 0:
        batch_size = 1  # Process one sample at a time when rotating
        
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    pred_label = torch.zeros_like(test_label)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            # Clear cache before processing each batch
            torch.cuda.empty_cache()
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # Get current batch data (keep on CPU initially)
            batch_data = test_data[start_idx:end_idx].clone()
            
            # Apply rotation on CPU if specified
            if rotation_angle != 0:
                # Process rotation on CPU
                for j in range(batch_data.shape[0]):
                    sample = batch_data[j]
                    batch_data[j] = torch.matmul(sample, rot_matrix)
            
            # Move data to device only for inference
            try:
                batch_data = batch_data.to(device)
                batch_pred = model(batch_data)
                pred_label[start_idx:end_idx] = torch.argmax(batch_pred, dim=2).cpu()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"GPU out of memory at batch {i}, falling back to CPU")
                    # Move model to CPU temporarily
                    model = model.cpu()
                    batch_pred = model(batch_data.cpu())
                    pred_label[start_idx:end_idx] = torch.argmax(batch_pred, dim=2)
                    # Move model back to GPU
                    model = model.to(device)
                else:
                    raise e
            
            # Clean up GPU memory
            del batch_data
            torch.cuda.empty_cache()
    
    # Calculate overall accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    
    # Calculate per-object accuracies
    object_accuracies = []
    for i in range(num_samples):
        acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / test_label[i].size(0)
        object_accuracies.append((i, acc))
    
    return test_accuracy, object_accuracies, pred_label


def test_rotation_range(model, test_data, test_label, device, batch_size=32, axis='z', output_dir='./robustness_results_seg'):
    """
    Tests segmentation accuracy across a range of rotation angles.
    
    Args:
        model: The segmentation model
        test_data: Tensor of test data
        test_label: Tensor of test labels
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        axis: Rotation axis ('x', 'y', or 'z')
        output_dir: Directory to save results
        
    Returns:
        angles: Array of angles tested
        accuracies: Array of corresponding accuracies
    """
    angles = np.arange(0, 360, 10)  # Test rotations every 30 degrees
    accuracies = []
    
    print(f"Testing rotation robustness around {axis}-axis...")
    for angle in angles:
        acc, _, _ = evaluate_model(
            model=model,
            test_data=test_data,
            test_label=test_label,
            device=device,
            batch_size=batch_size,
            rotation_angle=angle,
            rotation_axis=axis
        )
        accuracies.append(acc * 100)  # Convert to percentage
        print(f"  Rotation: {angle}° - Accuracy: {acc:.4f}")
    
    # Create results directory
    results_dir = f"{output_dir}/rotation_results"
    create_dir(results_dir)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(angles, accuracies, marker='o')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Segmentation Accuracy vs. Rotation Angle ({axis}-axis)')
    plt.grid(True)
    plt.savefig(f"{results_dir}/rotation_robustness_{axis}axis.png")
    plt.close()
    
    # Save results as CSV
    results = np.column_stack((angles, accuracies))
    np.savetxt(f"{results_dir}/rotation_robustness_{axis}axis.csv", results, delimiter=',', 
               header='angle,accuracy', comments='')
    
    return angles, accuracies


def test_point_sampling_range(model, test_data, test_label, device, batch_size=32, output_dir='./robustness_results_seg'):
    """
    Tests segmentation accuracy across a range of point counts.
    
    Args:
        model: The segmentation model
        test_data: Tensor of test data
        test_label: Tensor of test labels
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        output_dir: Directory to save results
        
    Returns:
        point_counts: Array of point counts tested
        accuracies: Array of corresponding accuracies
    """
    point_counts = [50, 100, 500, 1000, 5000, 10000]
    accuracies = []
    
    print("Testing point count robustness...")
    for num_points in point_counts:
        if num_points == 10000:
            # Use all points
            indices = None
        else:
            # Randomly sample points
            indices = np.random.choice(10000, num_points, replace=False)
        
        acc, _, _ = evaluate_model(
            model=model,
            test_data=test_data,
            test_label=test_label,
            device=device,
            batch_size=batch_size,
            point_indices=indices
        )
        accuracies.append(acc * 100)  # Convert to percentage
        print(f"  Points: {num_points} - Accuracy: {acc:.4f}")
    
    # Create results directory
    results_dir = f"{output_dir}/points_results"
    create_dir(results_dir)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(point_counts, accuracies, marker='o')
    plt.xlabel('Number of Points')
    plt.ylabel('Accuracy (%)')
    plt.title('Segmentation Accuracy vs. Number of Points')
    plt.grid(True)
    plt.savefig(f"{results_dir}/point_count_robustness.png")
    plt.close()
    
    # Save results as CSV
    results = np.column_stack((point_counts, accuracies))
    np.savetxt(f"{results_dir}/point_count_robustness.csv", results, delimiter=',', 
               header='num_points,accuracy', comments='')
    
    return point_counts, accuracies


def visualize_examples(test_data, test_label, pred_label, object_accuracies, output_dir, device, exp_name="exp"):
    """
    Visualizes the best and worst examples for analysis.
    
    Args:
        test_data: Test data tensor
        test_label: Ground truth labels
        pred_label: Predicted labels
        object_accuracies: List of (index, accuracy) pairs
        output_dir: Directory to save visualizations
        device: Device for visualization
        exp_name: Experiment name for file naming
    """
    # Sort by accuracy
    object_accuracies.sort(key=lambda x: x[1])
    
    # Create visualization directory
    create_dir(output_dir)
    
    # Visualize specified example if provided
    selected_idx = int(args.i)
    if selected_idx < len(test_data):
        object_accuracy = pred_label[selected_idx].eq(test_label[selected_idx].data).cpu().sum().item() / (test_label[selected_idx].reshape((-1,1)).size()[0])
        print(f"Accuracy of object {selected_idx}: {object_accuracy:.4f}")
        
        viz_seg(test_data[selected_idx], test_label[selected_idx], f"{output_dir}/gt_{selected_idx}.gif", device)
        viz_seg(test_data[selected_idx], pred_label[selected_idx], f"{output_dir}/pred_{selected_idx}_acc_{object_accuracy:.2f}.gif", device)
    
    # Visualize 2 worst predictions
    print("Visualizing worst predictions...")
    for i, acc in object_accuracies[:2]:
        viz_seg(test_data[i], test_label[i], f"{output_dir}/gt_bad_{i}.gif", device)
        viz_seg(test_data[i], pred_label[i], f"{output_dir}/pred_bad_{i}_acc_{acc:.2f}.gif", device)
        print(f"Bad prediction example {i}: accuracy = {acc:.4f}")
    
    # Visualize 3 good predictions
    print("Visualizing best predictions...")
    for i, acc in object_accuracies[-3:]:
        viz_seg(test_data[i], test_label[i], f"{output_dir}/gt_good_{i}.gif", device)
        viz_seg(test_data[i], pred_label[i], f"{output_dir}/pred_good_{i}_acc_{acc:.2f}.gif", device)
        print(f"Good prediction example {i}: accuracy = {acc:.4f}")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Initialize Model for Segmentation Task
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded checkpoint from {}".format(model_path))

    # Load test data and labels
    test_data = torch.from_numpy(np.load(args.test_data))
    test_label = torch.from_numpy(np.load(args.test_label))

    # Run specified experiment type
    if args.rotation_range:
        # Test across rotation angles
        angles, accuracies = test_rotation_range(
            model=model,
            test_data=test_data,
            test_label=test_label,
            device=args.device,
            batch_size=args.batch_size,
            axis=args.rotation_axis,
            output_dir=args.output_dir
        )
        print(f"Rotation robustness results saved to {args.output_dir}/rotation_results")
        
    elif args.point_range:
        # Test across point counts
        point_counts, accuracies = test_point_sampling_range(
            model=model,
            test_data=test_data,
            test_label=test_label,
            device=args.device,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        print(f"Point count robustness results saved to {args.output_dir}/points_results")
        
    else:
        # Standard evaluation with optional modifications
        # Sample points if num_points is specified
        if args.num_points < 10000:
            indices = np.random.choice(10000, args.num_points, replace=False)
        else:
            indices = None
            
        # Create output directory with experiment parameters
        experiment_dir = f"{args.output_dir}/np{args.num_points}_rot{args.rotation}{args.rotation_axis}"
        create_dir(experiment_dir)
        
        # Run evaluation
        test_accuracy, object_accuracies, pred_label = evaluate_model(
            model=model,
            test_data=test_data,
            test_label=test_label,
            device=args.device,
            batch_size=args.batch_size,
            point_indices=indices,
            rotation_angle=args.rotation,
            rotation_axis=args.rotation_axis
        )
        
        print(f"Test accuracy with {args.num_points} points and {args.rotation}° {args.rotation_axis}-axis rotation: {test_accuracy:.4f}")
        
        # Visualize examples
        visualize_examples(
            test_data=test_data if indices is None else test_data[:, indices, :],
            test_label=test_label if indices is None else test_label[:, indices],
            pred_label=pred_label,
            object_accuracies=object_accuracies,
            output_dir=experiment_dir,
            device=args.device,
            exp_name=args.exp_name
        )
        
        print("Evaluation complete.")