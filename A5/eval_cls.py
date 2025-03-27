import numpy as np
import argparse
import torch
import os
from models import cls_model
from data_loader import get_data_loader
from utils import create_dir, viz_cls

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

    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    
    return parser


def main(args):
    """
    Loads trained model and evaluates on test set.
    """
    # Create model directory
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir + "/" + args.task

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
    
    # Set up metrics
    correct_obj = 0
    num_obj = 0
    
    # Get class names
    class_names = ['chair', 'vase', 'lamp']
    
    # Visualization data
    viz_data = {
        'successes': {i: [] for i in range(len(class_names))},
        'failures': {i: [] for i in range(len(class_names))}
    }
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds.to(args.device)
            labels = labels.to(args.device).to(torch.long)
            
            # Make predictions
            predictions = model(point_clouds)
            pred_labels = torch.argmax(predictions, dim=1)
            
            # Collect visualization data
            for i in range(len(labels)):
                is_correct = pred_labels[i].item() == labels[i].item()
                true_label = labels[i].item()
                pred_label = pred_labels[i].item()
                
                if is_correct:
                    if len(viz_data['successes'][true_label]) < args.i:
                        viz_data['successes'][true_label].append({
                            'points': point_clouds[i].cpu().numpy(),
                            'true_label': true_label,
                            'pred_label': pred_label
                        })
                else:
                    if len(viz_data['failures'][true_label]) < args.i:
                        viz_data['failures'][true_label].append({
                            'points': point_clouds[i].cpu().numpy(),
                            'true_label': true_label,
                            'pred_label': pred_label
                        })
            
            # Update metrics
            correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
            num_obj += labels.size()[0]
    
    # Calculate accuracy
    accuracy = correct_obj / num_obj
    print("Test accuracy: {:.4f}".format(accuracy))
    
    # Create visualization directory
    create_dir('./visualization')
    
    # Visualize successful classifications
    print("Visualizing successful classifications...")
    for class_idx, class_name in enumerate(class_names):
        sample_data = viz_data['successes'][class_idx]
        for i, data in enumerate(sample_data):
            outfile = f'./visualization/success_{class_name}_{i}.gif'
            viz_cls(data['points'], data['true_label'], data['pred_label'], class_names, outfile)

    # Visualize failed classifications
    print("Visualizing failed classifications...")
    for class_idx, class_name in enumerate(class_names):
        sample_data = viz_data['failures'][class_idx]
        for i, data in enumerate(sample_data):
            outfile = f'./visualization/failure_{class_name}_{i}.gif'
            viz_cls(data['points'], data['true_label'], data['pred_label'], class_names, outfile)
    
    print("Visualization complete.")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)