import torchvision
from torchvision import models
resnet50 = models.resnet50() #pretrained=True 加载模型以及训练过的参数
print(resnet50)