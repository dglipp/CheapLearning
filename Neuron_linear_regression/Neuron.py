import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

batch_size = 10
n_batches = 100
n_features = 5
W = 0.01*np.ones((n_features))
b = np.random.rand(1)
lr = 1e-4
max_it = 1e2

w_real = np.random.rand(n_features) * 10 - 5
b_real = np.random.rand(1) * 10 - 5
X = np.random.rand(n_batches*batch_size, n_features) * 2 - 1
y = np.dot(X, w_real) + b_real + np.random.randn(n_batches*batch_size)

def forward(X, y):
    P = np.dot(X, W) + b
    L = np.sum(np.power(P-y, 2))
    return P, L

def backward(X, y, P):
    dldw = np.dot(X.T, -2 * (y-P))
    dldb = np.sum(-2 * (y-P))
    return dldw, dldb

loss = 100
it = 0
loss_vals = [2000, 1500]
v_loss_vals = [2000, 1500]
valid_size = 100
x_valid = np.random.rand(valid_size, n_features)*10 - 5
y_valid = np.dot(x_valid, w_real) + b_real
while it < max_it:
    it += 1
    for i in range(n_batches):
        X_batch = X[batch_size*i:batch_size*(i+1),:] 
        y_batch = y[batch_size*i:batch_size*(i+1)]
        pred, loss = forward(X_batch, y_batch)
        dldw, dldb = backward(X_batch, y_batch, pred)
        W -= lr*dldw
        b -= lr*dldb
    loss_vals.append(loss)
    _, v_loss = forward(x_valid, y_valid)
    v_loss_vals.append(v_loss)
    print("                                                                                ", end = "\r")
    print("epoch: " + str(it) + 
    "\tloss: " + str(int(loss)) +
    "\t\tvalid_loss: " + str(int(v_loss)), end = "\r")

loss_vals = loss_vals[2:]
v_loss_vals = v_loss_vals[2:]

print("\n")
if it > max_it:
    print("Max iterations exceeded")
print("w_real: " + str(w_real))
print("w_pred: " + str(W))
print("\n")
print("b_real: " + str(b_real))
print("b_pred: " + str(b))
print("\n")
s = np.sqrt(v_loss/valid_size)
print("rmse: " + str(s))
if n_features < 6:
    x_vals = np.repeat(np.linspace(-1, 1, 100), n_features).reshape((100, n_features)) 
    fig, ax = plt.subplots(n_features, 1)
    for i in range(n_features):
        ax[i].plot(x_vals[:,i], (np.dot(x_vals, W) + b), color = sns.color_palette(n_colors=n_features*3)[3*i], label = "predicted")
        ax[i].plot(X, y, "o", color = sns.color_palette(n_colors=n_features*3)[3*i+1], markersize=0.1)
        ax[i].plot(x_vals[:,i], (np.dot(x_vals, w_real) + b_real), color = sns.color_palette(n_colors=n_features*3)[3*i+2], label = "real")
        ax[i].set_title(str(i)+"-th component")
        ax[i].set_xlabel("x_"+str(i))
        ax[i].set_ylabel("y")
        ax[i].legend()

    fig.tight_layout(pad=0.1)

fig, ax = plt.subplots(1, 2)
ax[0].plot(loss_vals)
ax[0].set_title("train loss")
ax[1].plot(v_loss_vals)
ax[1].set_title("validation loss")
plt.show()