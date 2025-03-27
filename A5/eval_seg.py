import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


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
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])

    # ------ TO DO: Make Prediction ------
    batch_size = args.batch_size
    num_samples = test_data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    pred_label = torch.zeros_like(test_label)
    
    # Process data in batches to avoid memory issues
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_data = test_data[start_idx:end_idx].to(args.device)
            batch_pred = model(batch_data)
            pred_label[start_idx:end_idx] = torch.argmax(batch_pred, dim=2).cpu()
    
    # Calculate overall accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print("test accuracy: {}".format(test_accuracy))
    
    # Calculate per-object accuracy for selected index
    object_accuracy = pred_label[args.i].eq(test_label[args.i].data).cpu().sum().item() / (test_label[args.i].reshape((-1,1)).size()[0])
    print("accuracy of object {}: {:.4f}".format(args.i, object_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
    
    # Visualize additional examples (both good and bad predictions)
    # Compute per-object accuracy
    object_accuracies = []
    for i in range(min(100, num_samples)):  # Check first 100 objects to find good/bad examples
        acc = pred_label[i].eq(test_label[i].data).cpu().sum().item() / test_label[i].size(0)
        object_accuracies.append((i, acc))
    
    # Sort by accuracy
    object_accuracies.sort(key=lambda x: x[1])
    
    # Visualize 2 worst predictions
    for i, acc in object_accuracies[:2]:
        viz_seg(test_data[i], test_label[i], "{}/gt_bad_{}_acc_{:.2f}.gif".format(args.output_dir, i, acc), args.device)
        viz_seg(test_data[i], pred_label[i], "{}/pred_bad_{}_acc_{:.2f}.gif".format(args.output_dir, i, acc), args.device)
        print(f"Bad prediction example {i}: accuracy = {acc:.4f}")
    
    # Visualize 3 good predictions
    for i, acc in object_accuracies[-3:]:
        viz_seg(test_data[i], test_label[i], "{}/gt_good_{}_acc_{:.2f}.gif".format(args.output_dir, i, acc), args.device)
        viz_seg(test_data[i], pred_label[i], "{}/pred_good_{}_acc_{:.2f}.gif".format(args.output_dir, i, acc), args.device)
        print(f"Good prediction example {i}: accuracy = {acc:.4f}")