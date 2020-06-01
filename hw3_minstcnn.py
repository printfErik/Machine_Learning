import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import time

#define CNN
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv = nn.Conv2d(1,20,3,1)
    self.maxpool = nn.MaxPool2d(2,1)
    self.fc1 = nn.Linear(25*25*20,128)
    self.fc2 = nn.Linear(128,10)

  def forward(self,x):
    x = self.conv(x)
    x = self.maxpool(x)
    x = F.relu(x)
    x = x.view(-1,25*25*20)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.log_softmax(x,dim = 1)
    return x

# train our model, same procedure as fully connected layer
def train(model,data_set,optimizer):
  loss_p = 1e5
  loss_cur = 1
  tol = 1e-3 # set tolerance 
  epoc = 1
  loss_rate = []
  accuracy = []
  while (loss_p - loss_cur > tol or loss_cur > 0.05):
    train_acc = 0
    train_loss = 0
    model.train()   
    for data,label in data_set:
      output = model(data)
      loss = F.nll_loss(output,label)
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
    epoc+=1
  print("Training Done")
  return loss_rate, accuracy

# test the model we have trained
def test(model,test_set):
  acc_test = 0
  for data, label in test_set:
      output = model(data)
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
cnn = CNN()
opt = optim.SGD(cnn.parameters(),lr = 0.01)
loss, accuracy = train(cnn,train_loader,opt)

# save our model
torch.save(cnn, 'mnist-cnn')

# plot as line chart
plt.figure(1)
plt.plot(loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss_cnn.png")
plt.figure(2)
plt.plot(accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig("acc_cnn.png")
plt.show()

# test our modle on test set
trained_model = torch.load('mnist-cnn')
accuracy_test = test(trained_model,test_loader)
print("Testing accuracy is {:.2f}%".format(accuracy_test*100))


# for problem 6.4, we compare the influence for training time by different batch size
batch_size = [32,64,96,128]
train_time = []
loss_diff_batch = []
accuracy_diff_batch = []
batch_compare_cnn = CNN()
for i in range(len(batch_size)):
  # load different batch size data
  train_diff_batch = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size[i], shuffle = True)
  test_diff_batch = torch.utils.data.DataLoader(dataset = testset, batch_size = batch_size[i], shuffle = True)
  batch_compare_cnn.__init__()
  opt = optim.SGD(batch_compare_cnn.parameters(),lr = 0.01)
  # compute time
  start = time.time()
  loss, accuracy = train(batch_compare_cnn,train_diff_batch,opt)
  end = time.time()
  train_time.append(end-start)
  loss_diff_batch.append(loss)
  accuracy_diff_batch.append(accuracy)
  print("Batch size: {}, Training time: {:.2f}(s).".format(batch_size[i], end-start))

plt.figure()
plt.plot(batch_size,train_time,marker = "o")
plt.title('Run Time by Different Batch Size')
plt.xlabel('Batch size')
plt.ylabel('Run time')
plt.xticks(batch_size)
plt.savefig("time.png")
plt.show()

# for problem 6.5, we compare the influence for accuracy and loss by using different optimizer
cnn_SGD = CNN()
opt_SGD = optim.SGD(cnn_SGD.parameters(), lr=0.01)

cnn_ADAM = CNN()
opt_ADAM = optim.Adam(cnn_ADAM.parameters(), lr=0.01)

cnn_ADAGRAD = CNN()
opt_ADAGRAD = optim.Adagrad(cnn_ADAGRAD.parameters(), lr=0.01)

model_diff_opt = [cnn_SGD,cnn_ADAM,cnn_ADAGRAD]
diff_opt = [opt_SGD,opt_ADAM,opt_ADAGRAD]
loss_diff_opt = []
accuracy_diff_opt = []
for i in range(len(model_diff_opt)):
  loss, accuracy = train(model_diff_opt[i],train_loader,diff_opt[i])
  loss_diff_opt.append(loss)
  accuracy_diff_opt.append(accuracy)

plt.figure(1)
plt.plot(loss_diff_opt[0], label = "SGD")
plt.plot(loss_diff_opt[1], label = "ADAM")
plt.plot(loss_diff_opt[2], label = "ADAGRAD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_opt.png")
plt.figure(2)
plt.plot(accuracy_diff_opt[0], label = "SGD")
plt.plot(accuracy_diff_opt[1], label = "ADAM")
plt.plot(accuracy_diff_opt[2], label = "ADAGRAD")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc_opt.png")
plt.show()







