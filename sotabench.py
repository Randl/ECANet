import os

import torch
import torchvision.transforms as transforms
from torchbench.datasets.utils import download_file_from_google_drive
from torchbench.image_classification import ImageNet

import models

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Model 1
file_id = '1LMRFRTyzVifGBi2MUpTuYEWW44S8mwyl'
destination = './tmp/'
filename = 'eca_resnet18_k3577.pth.tar'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = models.__dict__['eca_resnet18'](k_size=[3, 5, 7, 7])
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ECA-Net18',
    paper_arxiv_id='1910.03151',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7092, 'Top 5 Accuracy': 0.8993},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 2
file_id = '15LV5Jkea3GPzvLP5__H7Gg88oNQUxBDE'
destination = './tmp/'
filename = 'eca_resnet34_k3357.pth.tar'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = models.__dict__['eca_resnet34'](k_size=[3, 3, 5, 7])
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ECA-Net34',
    paper_arxiv_id='1910.03151',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7421, 'Top 5 Accuracy': 0.9183},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 3
file_id = '1670rce333c_lyMWFzBlNZoVUvtxbCF_U'
destination = './tmp/'
filename = 'eca_resnet50_k3557.pth.tar'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = models.__dict__['eca_resnet50'](k_size=[3, 5, 5, 7])
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ECA-Net50',
    paper_arxiv_id='1910.03151',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7742, 'Top 5 Accuracy': 0.9362},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 4
file_id = '1b5FQ8yDFnZ_UhvWT9txmjI_LjbKkgnvC'
destination = './tmp/'
filename = 'eca_resnet101_k3357.pth.tar'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = models.__dict__['eca_resnet101'](k_size=[3, 3, 5, 7])
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ECA-Net101',
    paper_arxiv_id='1910.03151',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7865, 'Top 5 Accuracy': 0.9434},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 5
file_id = '1_bYnaOg9ptsILC_iC7uQ5Izv-u2rjYG5'
destination = './tmp/'
filename = 'eca_resnet152_k3357.pth.tar'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]

# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = models.__dict__['eca_resnet152'](k_size=[3, 3, 5, 7])
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ECA-Net152',
    paper_arxiv_id='1910.03151',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7892, 'Top 5 Accuracy': 0.9455},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 6
file_id = '1FxzeXPg1SJQZzVVH4HRjMeq_SVMfidUm'
destination = './tmp/'
filename = 'eca_mobilenetv2_k13.pth.tar'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = models.__dict__['eca_mobilenet_v2']()
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ECA-MobileNet_v2',
    paper_arxiv_id='1910.03151',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.7256, 'Top 5 Accuracy': 0.9081},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()
