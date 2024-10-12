import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

n = 1000; d = 100; lam = 0.1
X = np.vstack([np.random.normal(0.1, 1, (n//2, d)),
               np.random.normal(-0.1, 1, (n//2, d))])
Y = np.hstack([np.ones(n//2), -np.ones(n//2)])
w0 = np.random.normal(0, 1, d)


class lossfn:
    def __init__(self, X, Y, lam):
        self.X = X
        self.Y = Y
        self.lam = lam
    def __call__(self, w):
        return np.mean(np.maximum(0, 1 - self.Y * np.dot(self.X, w))) + self.lam *0.5* (np.linalg.norm(w))**2
    def subgrad(self, w):
        subgrad_values = np.array([np.zeros(x.shape) if y * np.dot(x, w) > 1 else -y * x for x, y in zip(self.X, self.Y)])
        mean_subgrad = np.mean(subgrad_values, axis=0)
        return mean_subgrad +  self.lam * w

class SVM:
    def __init__(self, X, Y, lossfn, w0, lr=0.1):
        self.lossfn = lossfn(X, Y, lam)
        self.w = w0
        self.lr = lr

    def step(self):
        self.w -= self.lr * self.lossfn.subgrad(self.w)

    def train(self, n_iter):
        loss = []
        accuracy = []
        for i in range(n_iter):
            self.step()
            loss.append(self.lossfn(self.w))
            accuracy.append(np.mean(np.sign(np.dot(X, self.w)) == Y))
        return loss, accuracy

    def predict(self, X):
        return np.sign(np.dot(X, self.w))
    
svm = SVM(X, Y, lossfn, w0)
loss, accuracy = svm.train(100)

plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("Loss value")  
plt.show()

plt.plot(accuracy)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
