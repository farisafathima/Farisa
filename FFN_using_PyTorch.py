#A simple feed forward network for a synthetic dataset created using PyTorch

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset




#create a synthetic dataset
class SyntheticDataset(Dataset):
    #custom dataset inherited from torch.utils.data.Dataset
    def __init__(self,num_samples=1000):
        super(SyntheticDataset,self).__init__()
        self.num_samples = num_samples
        self.data = torch.randn(num_samples,1,28,28) #random images
        self.labels = torch.randint(0,10,(num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.flatten =nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

train_data = SyntheticDataset(num_samples=1000)
test_data = SyntheticDataset(num_samples=200)
train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

model = SimpleNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)

        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if batch % 100 ==0 :
            current_samples = (batch + 1) * len(X)
            print(f"Loss: {loss.item():.6f} ")

#evaluation loop
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    total =0
    test_loss =0
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            correct += (pred.argmax(1)==y).sum().item()
            total+=y.size(0)
            loss = loss_fn(pred,y)
            test_loss+=loss.item()

    avg_test_loss = test_loss/len(dataloader)
    accuracy = 100 * correct /total

    print(f"Test loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")


epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader,model,loss_fn)
print("Training complete")

#save the model
torch.save(model.state_dict(),"model.pth")
print("Saved PyTorch model state to model.pth")
#loads the model
model = SimpleNN().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

#class labels for prediction
classes = [f"Class {i}" for i in range(10)]

model.eval()
x,y = test_data[0][0],test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x.unsqueeze(0))
    predicted,actual = classes[pred.argmax(1).item()], classes[y.item()]
    print(f"Predicted: {predicted}, actual : {actual}")


