from torch import nn
from fashion_trainer import FashionCNN




model = FashionCNN()
model = nn.DataParallel(model, device_ids=[0, 1])  # data parallel on gpu 0 and 1
preds = model(input)  # same as usual