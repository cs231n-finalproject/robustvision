import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from timm.models import create_model
from pathlib import Path
import os
from main import get_args_parser
from timm.models import create_model
from utils import load_pretrain_model
from visualization.net_visualization_pytorch import preprocess
from visualization.net_visualization_pytorch import compute_saliency_maps
from datasets import build_dataset
from main import main


def get_model_resnet101():
    # Download and load the pretrained resnet101 model.
    model_resnet101 = torchvision.models.resnet101(pretrained=True)

    # We don't want to train the model
    for param in model_resnet101.parameters():
        param.requires_grad = False

    return model_resnet101

def get_model_transconv(mode='eval', model='Transconv_small_patch16', device='cpu'): # or 'train'
    # Proposed Model
    parser = get_args_parser()
    args, unknowns = parser.parse_known_args()

    if mode == 'eval':
        args.return_model_eval = True
    else:
        args.return_model_train = True

    if model == 'Transconv_small_patch16':
        args.model = 'Transconv_small_patch16'
        args.resume = os.path.expanduser('~/Output/small/checkpoint.pth')
    else:
        args.model = 'Transconv_base_patch14'
        args.resume = os.path.expanduser('~/Output/base/checkpoint.pth')      

    args.device = device

    model_transconv = main(args)

    return model_transconv


def get_dataset():
    # Imagenet
    parser = get_args_parser()
    args, unknowns = parser.parse_known_args()
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    return dataset_train, dataset_val


def get_data_loader(dataset, batch_size=64):
    # Proposed Model
    parser = get_args_parser()
    args, unknowns = parser.parse_known_args()

    args.batch_size = batch_size
    data_loader_val = torch.utils.data.DataLoader(
        dataset, batch_size=int(3.0 * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    return data_loader_val


# from engine import evaluate

# model_transconv = get_model_transconv('eval', model='Transconv_small_patch16')

# _, dataset_val = get_dataset()
# data_loader_val = get_data_loader(dataset_val)
# test_stats = evaluate(data_loader_val, model_transconv, 'cuda')
# print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

# large_model_transconv = get_model_transconv(mode='eval', model='Transconv_base_patch14', device='cpu')
# test_stats_large = evaluate(data_loader_val, large_model_transconv, 'cpu')
# print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats_large['acc1']:.1f}%")


# resize = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])

# dataset_train, dataset_val = get_dataset()


# from visualization.data_utils import load_imagenet_val
# from vis_transconv import get_dataset
# dataset_train, dataset_val = get_dataset()
# X, y, class_names = load_imagenet_val(num=4)
# test_images = [Image.fromarray(x) for x in X]

# # add one more imagge
# def pil_loader(path: str) -> Image.Image:
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

# path = '/home/hawei/Dataset/ImageNet_ILSVRC2012/train/n02090622/n02090622_1914.JPEG'
# test_image = pil_loader(path)
# test_image = test_image.resize((224,224))
# test_images = [test_image] + test_images
# x = np.asarray(test_image)
# X = np.concatenate([x[None, :, :, :], X], axis=0)
# y = np.insert(y, 0, dataset_train.class_to_idx['n02090622'])

# plt.figure(figsize=(12, 6))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(X[i])
#     plt.title(class_names[y[i]])
#     plt.axis('off')
# plt.gcf().tight_layout()


# def show_saliency_maps(model, device='cpu'):
#     # Convert X and y from numpy arrays to Torch Tensors
#     X_tensor = torch.cat([preprocess(test_image) for test_image in test_images], dim=0).to(device)
#     y_tensor = torch.LongTensor(y).to(device)

#     # Compute saliency maps for images in X
#     saliency = compute_saliency_maps(X_tensor, y_tensor, model)

#     # Convert the saliency map from Torch Tensor to numpy array and show images
#     # and saliency maps together.
#     saliency = saliency.numpy()
#     N = X.shape[0]
#     for i in range(N):
#         plt.subplot(2, N, i + 1)
#         plt.imshow(X[i])
#         plt.axis('off')
#         plt.title(class_names[y[i]])
#         plt.subplot(2, N, N + i + 1)
#         plt.imshow(saliency[i], cmap=plt.cm.hot)
#         plt.axis('off')
#         plt.gcf().set_size_inches(12, 5)
#     plt.show()

# # show_saliency_maps(get_model_resnet101())

# show_saliency_maps(model_transconv)
