import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

x, label = dataset[0]

print('size:', len(dataset)) # 60000
print('label:', label) # 5
print('type:', type(x)) # <class 'PIL.Image.Image'>
print('shape:', x.shape) # torch.Size([1, 28, 28])