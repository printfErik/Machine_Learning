import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

#define fully connected network
class Fully_Connected(nn.Module):
  def __init__(self):
    super(Fully_Connected, self).__init__()
    # Two fc layers
    self.fc1 = nn.Linear(28*28,128)
    self.fc2 = nn.Linear(128,10)

  def forward(self,x):
    x = x.view(-1,28*28)
    x = F.relu(self.fc1(x))
    x = F.log_softmax(self.fc2(x),dim = 1)
    return x

# training model
def train(model,data_set,optimizer):
  loss_p = 1e5
  loss_cur = 1
  tol = 1e-3
  epoc = 1
  loss_rate = []
  accuracy = []
  # test if converges
  while (loss_p - loss_cur > tol or loss_cur > 0.05):
    train_acc = 0
    train_loss = 0
    model.train()   
    for data,label in data_set:
      output = model(data.view(-1,784))
      loss = F.nll_loss(output,label) # cross-entropy
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      _, pred = output.max(1)
      correct = (pred == label).sum().item()
      acc = correct / data.shape[0]
      train_acc += acc
    loss_cur = train_loss/ len(data_set)
    acc_cur = train_acc/ len(data_set)
    loss_rate.append(loss_cur)
    accuracy.append(acc_cur)
    print("Epoch: {}, Loss: {:.6f}, Accuracy: {:.6f}".format(epoc, loss_cur, acc_cur))
    loss_p = loss_cur
    epoc += 1
  print("Training Done.")
  return loss_rate, accuracy

# test the model we have trained
def test(model,test_set):
  acc_test = 0
  for data, label in test_set:
      output = model(data.view(-1,784))
      _, pred = output.max(1)
      correct = (pred == label).sum().item()
      acc = correct / data.shape[0]
      acc_test += acc
  accuracy = acc_test/len(test_set)
  return accuracy



# load data
trainset = torchvision.datasets.MNIST(root='./data',train = True, download = True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size = 32, shuffle = True)

# create object
FC = Fully_Connected()
opt = optim.SGD(FC.parameters(),lr = 0.01) #SDG optimizer
loss_rate, accuracy = train(FC,train_loader,opt)

# save our model
torch.save(FC, 'mnist-fc')

# plot as line chart
plt.figure(1)
plt.plot(loss_rate)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss.png")
plt.figure(2)
plt.plot(accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("acc.png")
plt.show()

# test our modle on test set
trained_model = torch.load('mnist-fc')
accuracy_test = test(trained_model,test_loader)
print("Testing accuracy is {:.2f}%".format(accuracy_test*100))