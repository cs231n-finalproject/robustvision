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


def get_model_resnet101():
    # Download and load the pretrained resnet101 model.
    model_resnet101 = torchvision.models.resnet101(pretrained=True)

    # We don't want to train the model
    for param in model_resnet101.parameters():
        param.requires_grad = False

    return model_resnet101

def get_model_transconv():
    # Proposed Model
    parser = get_args_parser()
    args, unknowns = parser.parse_known_args()

    print(f"Creating model: {args.model}")
    transconv_model = create_model(
        args.model,
        pretrained=args.pretrained,
        finetune_vit=args.finetune_vit,
        finetune_conv=args.finetune_conv,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
    )
    checkpoint_paths = args.resume if args.resume else '~/Output/small/checkpoint.pth'
    model_transconv = load_pretrain_model(transconv_model, checkpoint_paths, finetune=False)
    # We don't want to train the model
    # for param in model_transconv.parameters():
    #     param.requires_grad = False

    return model_transconv


def get_dataset():
    # Imagenet
    parser = get_args_parser()
    args, unknowns = parser.parse_known_args()
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    return dataset_train, dataset_val


resize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset_train, dataset_val = get_dataset()


from visualization.data_utils import load_imagenet_val
X, y, class_names = load_imagenet_val(num=5)
# np.append(y, '703')
test_images = [Image.fromarray(x) for x in X]

from visualization.data_utils import pil_loader
path = '/home/hawei/Dataset/ImageNet_ILSVRC2012_FULL/train/n03891251/n03891251_6013.JPEG'
test_image = pil_loader(path)
test_image = test_image.resize((224,224))
test_images += [test_image]
x = np.asarray(test_image)
X = np.concatenate([x[None, :, :, :], X], axis=0)
y = np.insert(y, 0, 703)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i])
    plt.title(class_names[y[i]])
    plt.axis('off')
plt.gcf().tight_layout()


def show_saliency_maps(model):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(test_image) for test_image in test_images], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

show_saliency_maps(get_model_resnet101())

show_saliency_maps(get_model_transconv())
