import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image 


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: MLP-Mixer: An all-MLP Architecture for Vision""")
    home_dir = os.getcwd()
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--model-folder", default='{}/model/mlp/'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=300, type=int)
    
    args = parser.parse_args()
    print('---------------------Welcome to MLP Mixer-------------------')
    print('---------------------------------------------------------------------')
    print('Predict using MLP Mixer for image path: {}'.format(args.test_file_path))
    print('===========================')
    
    return args

def predict(opt):
    model = torch.load(opt.model_folder)
    model.eval()

    # Load test images from folder
    image = Image.open(opt.test_file_path).convert('RGB')
    trans= transforms.Compose([transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor()])
    image_tensor = trans(image)
    input_batch = image_tensor.unsqueeze(0)
    output = model(input_batch)
    # Get predicted class
    _, predicted_class = torch.max(output, 1)
    print(f"Predicted class index: {predicted_class}")

if __name__ == "__main__":
    opt = get_args()
    predict(opt)

# python predict.py --test-file-path data/valid/cats/cat.2000.jpg --model-folder model/mlp/e_4 --image-size 300