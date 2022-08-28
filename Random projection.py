# import libraries
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from customize_data import CustomData
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Build a basic neural network model with one hidden layer
class NN(nn.Module):
  def __init__(self,input_size,num_classes):
    super(NN,self).__init__()
    self.fc1 = nn.Linear(input_size,50)
    self.fc2 = nn.Linear(50,num_classes)

  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# load mnist dataset from torch.datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data',train=False,transform = transforms.ToTensor(),download=True)


# Vectorize the data along the first dimension
train = train_dataset.data.reshape(-1, train_dataset.data.shape[1]*train_dataset.data.shape[2])
test = test_dataset.data.reshape(-1, test_dataset.data.shape[1]*test_dataset.data.shape[2])


train_dataset = CustomData(train, train_dataset.targets)
test_dataset = CustomData(test, test_dataset.targets)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)


# Define random projection matrxi
random_projection =  torch.randn(784,100)

# set hyperparameters
input_size = random_projection.shape[1]
num_classes = 10
learning_rate = 0.001
epochs = 10


# Initialize model
model = NN(input_size=input_size,num_classes=num_classes).to(device)

# Define Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

# train the model
for i in range(epochs):
  for _,(data,label) in enumerate(train_loader):

    data = data.reshape(data.shape[0],-1)
    data = torch.matmul(data,random_projection)
    data = data.float().to(device)
    label = label.to(device)

    # forward pass
    out = model(data)
    loss = loss_fn(out,label)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0],-1)
            x=  torch.matmul(x,random_projection)
            x = x.float().to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

print(check_accuracy(train_loader,model))
print(check_accuracy(test_loader,model))
