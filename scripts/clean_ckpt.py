import torch
import argparse

def remove_checkpoint_keys(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    for k in list(checkpoint.keys()):
        if k != "model" and k != "config":
            del checkpoint[k]

    # Save the modified checkpoint
    torch.save(checkpoint, checkpoint_path)

if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Remove everything but the state_dict keys from a checkpoint')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    args = parser.parse_args()

    # Remove the checkpoint keys
    remove_checkpoint_keys(args.checkpoint_path)