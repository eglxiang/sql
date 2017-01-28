# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np 
import argparse
import pdb

def sigmoid_activation(x):
	return 1.0/(1+np.exp(-x))

def next_batch(X, y, batchSize):
	for i in np.arange(0, X.shape[0], batchSize):
		yield(X[i:i+batchSize], y[i:i+batchSize])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="size pf SGD mini-batches")
args = vars(ap.parse_args())

# generate a 2-class clssification problem with 400 data points
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)

X = np.c_[np.ones((X.shape[0])),X]

print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

lossHistory = []

# loop over the desired num of epochs
for epoch in np.arange(0, args["epochs"]):
	# initialize the total loss for the epoch
	epochLoss = []

	# loop over our data in batches
	for (batchX, batchY) in next_batch(X,y,args["batch_size"]):
		preds = sigmoid_activation(batchX.dot(W))
		error = preds - batchY
		loss = np.sum(error**2)
		epochLoss.append(loss)
		gradient = batchX.T.dot(error) / batchX.shape[0]
		W += -args["alpha"] * gradient

	# update our loss history list by taking the average loss across all frames
	lossHistory.append(np.average(epochLoss))
 	
# compute the line of the best fit by setting the sigmoid function to 0
# and solving for X2 in terms of X1
#pdb.set_trace()
Y = (-W[0] - (W[1]*X)) / W[2]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X[:,1], X[:,2], marker="o", c=y)
plt.plot(X, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()