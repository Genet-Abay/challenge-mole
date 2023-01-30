import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)



print(train_data.data.size())
print(train_data.targets.size())