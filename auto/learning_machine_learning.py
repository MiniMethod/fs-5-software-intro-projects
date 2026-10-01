import matplotlib.pyplot as plt
import torch
from torch import nn

# part 0 and 1 https://www.learnpytorch.io/01_pytorch_workflow/

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device used: " + device)

#weights
weight = 50

start = 0
end = 2
step = 0.02

#creates the data set
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X

#shows a bit of data
print(X[:10])
print(y[:10])

#splits the data for training/testing -- validation usually here as well...
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"X_train len: {len(X_train)}, y_train len: {len(y_train)}, X_test len: {len(X_test)}, y_test len: {len(y_test)}")

#plots the data 
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
	plt.figure(figsize=(10, 7))

	plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")

	plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

	if predictions is not None:
		plt.scatter(test_data, predictions, c="r", s=4, label="Predictions" )

	plt.show()

# plot_predictions()

# Create a Linear Regression model class
class LinearRegressionModel(nn.Module):
	def __init__(self):
		super().__init__() 
		self.linear_layer = nn.Linear(in_features=1, out_features=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.linear_layer(x)

torch.manual_seed(1)

model = LinearRegressionModel()

print(next(model.parameters()).device)
model.to(device)
print(next(model.parameters()).device)

#have to load data to gpu
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

print(model.state_dict())

y_preds = None

with torch.inference_mode(): 
	y_preds = model(X_test)

print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds.cpu())

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 15000

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
	# Training

	# Put model in training mode (this is the default state of a model)
	model.train()

	# 1. Forward pass on train data using the forward() method inside 
	y_pred = model(X_train)
	# print(y_pred)

	# 2. Calculate the loss (how different are our models predictions to the ground truth)
	loss = loss_fn(y_pred, y_train)

	# 3. Zero grad of the optimizer
	optimizer.zero_grad()

	# 4. Loss backwards
	loss.backward()

	# 5. Progress the optimizer
	optimizer.step()

	# Testing

	# Put the model in evaluation mode
	model.eval()

	with torch.inference_mode():
	# 1. Forward pass on test data
		test_pred = model(X_test)

		# 2. Caculate loss on test data
		test_loss = loss_fn(test_pred, y_test) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

		if epoch % 100 == 0:
			print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

model.eval()

with torch.inference_mode():
	y_preds = model(X_test)

plot_predictions(predictions=y_preds.cpu())