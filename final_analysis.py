import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def get_data(file, label_col, ratio, bias, random_state = 0):
    
    df = pd.read_csv(file, index_col='PolicyNumber')
    
    data_true = df[df[label_col] == 1].copy()
    data_false = df[df[label_col] == 0].copy()

    num_rows_true = len(data_true)
    num_rows_false = len(data_false)

    num_sample_true = int(num_rows_true * ratio)
    num_sample_false = int(num_rows_true - num_sample_true) 

    all_idx_true = np.arange(0, num_rows_true)
    all_idx_false = np.arange(0, num_rows_false)

    rng = np.random.RandomState(random_state)
    
    train_sample_idx_true = rng.choice(all_idx_true, size = num_sample_true, replace = False)
    train_sample_idx_false = rng.choice(all_idx_false, size = num_sample_false, replace = False)

    test_sample_idx_true = np.setdiff1d(all_idx_true, train_sample_idx_true)
    test_sample_idx_false = np.setdiff1d(all_idx_false, train_sample_idx_false)

    train_sample_true = data_true.iloc[train_sample_idx_true].copy()
    train_sample_false = data_false.iloc[train_sample_idx_false].copy()

    test_sample_true = data_true.iloc[test_sample_idx_true].to_numpy()
    test_sample_false = data_false.iloc[test_sample_idx_false].to_numpy()
    
    train = np.vstack((train_sample_true, train_sample_false))
    test = np.vstack((test_sample_true, test_sample_false))
    
    np.random.shuffle(train)
    np.random.shuffle(test)
    
    
    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)
    
    df_train.to_csv(fr"datasets\train.csv", header = False, index = False)
    df_test.to_csv(fr"datasets\test.csv", header = False, index = False)
    
    
    # train.tofile('train.csv', sep = ',')
    # test.tofile('test.csv', sep = ',')
    
    train_x = train[:,:-1]
    train_x = np.hstack((train_x,np.ones((train_x.shape[0],1))))
    
    train_y = train[:, -1]
    
    test_x = test[:, :-1]
    test_x = np.hstack((test_x,np.ones((test_x.shape[0],1))))
    
    test_y = test[:, -1]
    
    return train_x, train_y, test_x, test_y

def confusion(true, pred, scale = 'all', n_round = 4, rates = 'all', data_name = ""):
    
    con = confusion_matrix(true, pred, normalize = scale)
    
    print(f"{data_name} Confusion Matrix: \n{np.round(con, 4)}\n")
    
    true_neg, false_pos, false_neg, true_pos = con.ravel()
    
    if rates == 'all':
        return true_pos, true_neg, false_pos, false_neg, 
    elif rates == 'true':
        return true_pos, true_neg
    else:
        return false_pos, false_neg
    
# Hyperparameters

max_epoch       = 200
train_batch_size= 923
test_batch_size = 2071*7
num_batches = 1
log_iter        = 100

TOLERANCE    = 1e-5
input_dim    = 65
n_class      = 2
SAMPLES      = 100
TEST_SAMPLES = 100

device = torch.device("cpu")

lr         = 0.001
n_hidden   = 20
activation = 'relu'

def get_data2(file):
    data1 = pd.read_csv(file, header=None).to_numpy()
    data_x = data1[:, 0:-1]
    data_x = np.hstack((data_x, np.ones((data_x.shape[0], 1))))

    data_y = data1[:, -1]
    
    return data_x, data_y

class Data(Dataset):

	def __init__(self, filename):
		x, y = get_data2(filename)
		self.X, self.Y = x, y

	def __getitem__(self, index):

		x = torch.tensor(self.X[index]).float()
		y = torch.tensor(self.Y[index]).long()
		return (x, y)

	def __len__(self):
		return self.X.shape[0]

class Gaussian(nn.Module):
	def __init__(self, mu, rho):
		super().__init__()
		self.mu     = mu
		self.rho    = rho
		self.normal = torch.distributions.Normal(0, 1)

	@property
	def sigma(self):
		return (torch.log(torch.exp(self.rho))+TOLERANCE)

	def sample(self):
		epsilon = self.normal.sample(self.rho.shape).type(self.mu.type()).to(device)
		return self.mu + self.sigma * epsilon

	def log_prob(self, input):
		return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class StandardGaussian(nn.Module):
	def __init__(self, sigma):
		super().__init__()
		self.sigma = sigma
		self.gaussian = torch.distributions.Normal(0, self.sigma)

	def log_prob(self, input):
		return (self.gaussian.log_prob(input)).sum()

class BayesianLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features

		alpha = 1.0 / np.sqrt(self.in_features)

		self.weight_mu  = nn.Parameter(torch.zeros(out_features, in_features))
		self.weight_rho = nn.Parameter(torch.ones(out_features, in_features)*alpha)
		self.weight     = Gaussian(self.weight_mu, self.weight_rho)

		self.bias_mu    = nn.Parameter(torch.zeros(out_features))
		self.bias_rho   = nn.Parameter(torch.ones(out_features)*alpha)
		self.bias       = Gaussian(self.bias_mu, self.bias_rho)

		self.weight_prior = StandardGaussian(1)
		self.bias_prior = StandardGaussian(1)
		self.log_prior = 0
		self.log_variational_posterior = 0

	def forward(self, input, sample=False, calculate_log_probs=False):
		if self.training or sample:
			weight = self.weight.sample()
			bias = self.bias.sample()
		else:
			weight = self.weight.mu
			bias = self.bias.mu
		if self.training or calculate_log_probs:
			self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
			self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
		else:
			self.log_prior, self.log_variational_posterior = 0, 0

		return F.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
	def __init__(self, n_hidden, activation):
		super().__init__()
		self.l1 = BayesianLinear(input_dim, n_hidden)
		self.l2 = BayesianLinear(n_hidden, n_hidden)
		self.l3 = BayesianLinear(n_hidden, n_class)
		if(activation=='relu'):
			self.activation_fn = nn.ReLU()
		else:
			self.activation_fn = nn.Tanh()

	def forward(self, x, sample=False):
		x = self.activation_fn(self.l1(x, sample))
		x = self.activation_fn(self.l2(x, sample))
		x = F.log_softmax(self.l3(x, sample), dim=1)
		return x

	def log_prior(self):
		return self.l1.log_prior \
			   + self.l2.log_prior \
			   + self.l3.log_prior

	def log_variational_posterior(self):
		return self.l1.log_variational_posterior \
			   + self.l2.log_variational_posterior \
			   + self.l3.log_variational_posterior

	def sample_elbo(self, input, target, samples=SAMPLES):

		outputs = torch.zeros(samples, input.shape[0], n_class).to(device)
		log_priors = torch.zeros(samples).to(device)
		log_variational_posteriors = torch.zeros(samples).to(device)
		for i in range(samples):
			outputs[i] = self.forward(input, sample=True)
			log_priors[i] = self.log_prior()
			log_variational_posteriors[i] = self.log_variational_posterior()
		log_prior = log_priors.mean()
		log_variational_posterior = log_variational_posteriors.mean()
		outputs = outputs.mean(dim= 0)
		neg_log_like = F.nll_loss(outputs, target, reduction='mean')
		loss = neg_log_like
		pred = outputs.argmax(dim= 1)
		train_acc = ((pred.eq(target.view_as(pred)).sum())*100.0)/target.shape[0]
		return loss, train_acc

def train(net, train_loader, optimizer, epoch):
	net.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target   = data.to(device), target.to(device)
		net.zero_grad()
		loss, train_acc = net.sample_elbo(data, target, samples=SAMPLES)
		loss.backward()
		optimizer.step()  

def predictive_accuracy(net, test_loader):
	net.eval()
 
	preds = np.array([])
	targets = np.array([])
 
	with torch.no_grad():
		test_acc = 0
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target   = data.to(device), target.to(device)
			outputs = torch.zeros(TEST_SAMPLES, data.shape[0], n_class).to(device)
			for i in range(TEST_SAMPLES):
				outputs[i] = net(data, sample=True)
				pred = outputs[i].argmax(dim=1)

				preds = np.append(preds, pred.numpy().copy())
				targets = np.append(targets, target.numpy().copy())
    
				test_acc += ((pred.eq(target.view_as(pred)).sum()) / num_batches) / target.shape[0]
		test_acc = test_acc/TEST_SAMPLES
	return test_acc, preds, targets

def predictive_log_likelihood(net, test_loader):
	net.eval()
	with torch.no_grad():
		pred_like = 0
		counter = 0

		for batch_idx, (data, target) in enumerate(test_loader):
			data, target   = data.to(device), target.float().to(device)
			outputs = torch.zeros(TEST_SAMPLES, data.shape[0], n_class).to(device)
			for i in range(TEST_SAMPLES):
				outputs[i] = net(data, sample=True)
			output = torch.sum(outputs, axis = 0)/TEST_SAMPLES
			pred_like += torch.mean(target*output[:,1] + (1-target)*output[:,0])
			counter+=1
		pred_like = pred_like/counter
	return pred_like



X_train, y_train, X_test, y_test = get_data(fr'datasets\fraud_processed.csv', 'Fraud', .35, True, 0)

model = LogisticRegression(max_iter=1000, fit_intercept=False)

model.fit(X_train, y_train)
preds1 = model.predict(X_train)
preds2 = model.predict(X_test)

print(f"Test Accuracy: {np.round(np.mean(preds2 == y_test), 4)}\n")

test_tp, test_tn, test_fp, test_fn = confusion(y_test, preds2, scale = 'all', data_name = "Test")



network = nn.Sequential(
    nn.Linear(65, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)

X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_ts = torch.tensor(X_test, dtype=torch.float32)
y_ts = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

n_epochs = 100
batch_size = 71
rate = .01

loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(network.parameters(), lr=rate)

for epoch in range(n_epochs):
    
    shuffle_idx = torch.randperm(X_tr.size()[0])

    X_shuffled = X_tr[shuffle_idx].clone().detach()
    y_shuffled = y_tr[shuffle_idx].clone().detach()
    
    
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_shuffled[i:i+batch_size]
        y_pred = network(Xbatch)
        ybatch = y_shuffled[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

with torch.no_grad():
    y_pred = network(X_tr)

train_preds = y_pred.numpy()

train_preds[train_preds >= .5] = 1
train_preds[train_preds < .5] = 0

with torch.no_grad():
    y_pred2 = network(X_ts)

test_preds = y_pred2.numpy()

test_preds[test_preds >= .5] = 1
test_preds[test_preds < .5] = 0

test_accuracy = np.mean(test_preds == y_test)
print(f"Test Accuracy {test_accuracy}")

train_tp3, train_tn3, train_fp3, train_fn3 = confusion(y_test, test_preds, scale = 'all', data_name = 'Test')

train_data = Data(fr'datasets\train.csv')
test_data = Data(fr'datasets\test.csv')

train_load = DataLoader(dataset= train_data,  batch_size= train_batch_size, shuffle= True)
test_load = DataLoader(dataset= test_data,  batch_size= test_batch_size, shuffle= False)

net = BayesianNetwork(n_hidden= n_hidden, activation= activation).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = lr)

print(f"Hidden Layer Size : {n_hidden} \tActivation Function : {activation}\n")

train_like = []
test_like = []

for epoch in range(max_epoch):
    train(net, train_load, optimizer, epoch)
    train_like.append(predictive_log_likelihood(net, train_load))
    test_like.append(predictive_log_likelihood(net, test_load))

pred_a, predicts, trues = predictive_accuracy(net, test_load)
pred_like = predictive_log_likelihood(net, test_load)

print(f"Accuracy: {pred_a}")
print(f"Likelihood: {pred_like}")

train_tp2, train_tn2, train_fp2, train_fn2 = confusion(trues, predicts, scale = 'all', data_name = 'Test')

plt.plot(np.arange(max_epoch) + 1, train_like)
plt.title("Train")
plt.xlabel("Epoch")
plt.ylabel("Log-Likelihood")
plt.savefig('plot.png')
plt.close()