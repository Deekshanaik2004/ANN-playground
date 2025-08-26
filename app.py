
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

# ---------- Page setup ----------
st.set_page_config(page_title="Neural Network Playground", layout="wide")
st.title("🧠 Customizable Neural Network Playground")
st.caption("Play with a tiny MLP you can actually *see* learn. Choose data, tweak layers, and watch the decision boundary and loss change.")

# ---------- Utilities: MLP from scratch (NumPy) ----------
def init_params(layer_sizes, hidden_activation='relu', seed=42):
    rng = np.random.default_rng(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        if i < len(layer_sizes) - 2:
            # hidden layer
            if hidden_activation.lower() == 'relu':
                scale = np.sqrt(2.0 / n_in)  # He
            else:
                scale = np.sqrt(1.0 / n_in)  # Xavier (tanh/sigmoid)
        else:
            # output sigmoid
            scale = np.sqrt(1.0 / n_in)
        W = rng.normal(0, scale, size=(n_in, n_out))
        b = np.zeros((1, n_out))
        params.append({'W': W, 'b': b})
    return params

def act_forward(Z, activation):
    if activation == 'relu':
        return np.maximum(0, Z)
    elif activation == 'tanh':
        return np.tanh(Z)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    else:
        raise ValueError("Unknown activation")

def act_backward(dA, Z, activation):
    if activation == 'relu':
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
    elif activation == 'tanh':
        return dA * (1 - np.tanh(Z) ** 2)
    elif activation == 'sigmoid':
        s = 1 / (1 + np.exp(-Z))
        return dA * s * (1 - s)

def forward(X, params, hidden_activation='relu'):
    A = X
    caches = []
    L = len(params)
    for l in range(L):
        W = params[l]['W']; b = params[l]['b']
        A_prev = A
        Z = A_prev @ W + b
        if l == L - 1:
            A = 1 / (1 + np.exp(-Z))  # sigmoid
            activation = 'sigmoid'
        else:
            A = act_forward(Z, hidden_activation)
            activation = hidden_activation
        caches.append({'A_prev': A_prev, 'Z': Z, 'W': W, 'b': b, 'A': A, 'activation': activation})
    return A, caches

def compute_loss(y_true, y_hat, eps=1e-9):
    y_true = y_true.reshape(-1, 1)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)).mean()

def backward(y_true, caches):
    grads = [None] * len(caches)
    m = y_true.shape[0]
    y_true = y_true.reshape(-1, 1)
    dA = None
    for l in reversed(range(len(caches))):
        cache = caches[l]
        A_prev, Z, W, b, A, activation = cache['A_prev'], cache['Z'], cache['W'], cache['b'], cache['A'], cache['activation']
        if activation == 'sigmoid' and l == len(caches) - 1:
            dZ = A - y_true  # BCE + sigmoid
        else:
            dZ = act_backward(dA, Z, activation)
        dW = (A_prev.T @ dZ) / m
        db = dZ.mean(axis=0, keepdims=True)
        dA = dZ @ W.T
        grads[l] = {'dW': dW, 'db': db}
    return grads

def update_params(params, grads, lr=0.1):
    for l in range(len(params)):
        params[l]['W'] -= lr * grads[l]['dW']
        params[l]['b'] -= lr * grads[l]['db']

def predict(X, params, hidden_activation='relu'):
    y_hat, _ = forward(X, params, hidden_activation)
    return (y_hat >= 0.5).astype(int)

def decision_boundary_grid(X, params, hidden_activation, padding=0.6, step=0.02):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    yhat, _ = forward(grid, params, hidden_activation)
    Z = yhat.reshape(xx.shape)
    return xx, yy, Z

# ---------- Sidebar: Controls ----------
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Dataset")
    dataset = st.selectbox("Type", ["moons", "circles", "blobs (2 classes)"])
    n_samples = st.slider("Samples", 100, 2000, 400, step=50)
    noise = st.slider("Noise", 0.0, 0.5, 0.2, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    st.subheader("Model")
    hidden_activation = st.selectbox("Hidden activation", ["relu", "tanh", "sigmoid"])
    n_hidden = st.slider("Hidden layers", 1, 4, 2)
    width = st.slider("Neurons per hidden layer", 2, 64, 16)

    st.subheader("Training")
    lr = st.select_slider("Learning rate", options=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1], value=1e-1, format_func=lambda x: f"{x:g}")
    epochs = st.slider("Epochs", 50, 3000, 800, step=50)

    colb1, colb2 = st.columns(2)
    gen_clicked = colb1.button("🔀 Regenerate Data")
    train_clicked = colb2.button("🚀 Train / Retrain")

# ---------- Data generation ----------
def get_data(dataset, n_samples, noise, seed):
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    else:  # blobs
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0 + noise * 2, random_state=seed)
    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

if "X" not in st.session_state or gen_clicked:
    X, y = get_data(dataset, n_samples, noise, seed)
    st.session_state.X, st.session_state.y = X, y
else:
    X, y = st.session_state.X, st.session_state.y

# show raw data scatter
col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Data preview")
    fig_data, axd = plt.subplots()
    axd.scatter(X[:, 0], X[:, 1], c=y, s=12, alpha=0.8)
    axd.set_xlabel("x1"); axd.set_ylabel("x2"); axd.set_title("Dataset")
    st.pyplot(fig_data)

# ---------- Initialize / train model ----------
input_dim = 2
layer_sizes = [input_dim] + [width] * n_hidden + [1]

if "params" not in st.session_state or train_clicked or gen_clicked:
    st.session_state.params = init_params(layer_sizes, hidden_activation=hidden_activation, seed=seed)
    st.session_state.losses = []

# Train when button is clicked
if train_clicked:
    params = st.session_state.params
    losses = []
    progress = st.progress(0)
    for epoch in range(1, epochs + 1):
        y_hat, caches = forward(X, params, hidden_activation=hidden_activation)
        loss = compute_loss(y, y_hat)
        grads = backward(y, caches)
        update_params(params, grads, lr=lr)
        losses.append(loss)
        if epoch % max(1, epochs // 100) == 0:
            progress.progress(epoch / epochs)
    st.session_state.params = params
    st.session_state.losses = losses

# ---------- Visualizations ----------
params = st.session_state.params
losses = st.session_state.losses

with col2:
    st.subheader("Training loss")
    if losses:
        fig_loss, axl = plt.subplots()
        axl.plot(np.arange(1, len(losses) + 1), losses)
        axl.set_xlabel("Epoch"); axl.set_ylabel("Binary cross-entropy")
        axl.set_title("Loss curve")
        st.pyplot(fig_loss)
    else:
        st.info("Click **Train / Retrain** to see the loss curve.")

st.divider()
st.subheader("Decision boundary")
if losses:
    colA, colB = st.columns([1.2, 1])
    with colA:
        xx, yy, Z = decision_boundary_grid(X, params, hidden_activation)
        fig_db, ax = plt.subplots()
        cs = ax.contourf(xx, yy, Z, levels=25, alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=12, alpha=0.9, edgecolors='none')
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title("Decision boundary (probabilities)")
        st.pyplot(fig_db)
    with colB:
        y_pred = predict(X, params, hidden_activation).flatten()
        acc = (y_pred == y).mean()
        st.metric("Accuracy on training set", f"{acc*100:.1f}%")
        st.write("**Model size**")
        total_params = int(sum(p['W'].size + p['b'].size for p in params))
        st.write(f"{len(layer_sizes)-2} hidden layers × {width} neurons → **{total_params}** parameters")

        st.write("**Tip:** If the boundary looks noisy or training is unstable, try a smaller learning rate or more epochs. For complex shapes, increase hidden layers/neurons.")
else:
    st.info("Train the model to see the decision boundary.")

