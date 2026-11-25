import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    w = np.zeros((X.shape[1]))
    b = 0

    for step in range(steps):

        y_pred = _sigmoid(X @ w + b)

        eps = 1e-8
        safe_y_pred = np.clip(y_pred, eps, 1-eps)

        loss = -np.average(y * np.log(safe_y_pred) + (1-y) * np.log(1 - safe_y_pred))
    
        grad_w = X.T @ (y_pred - y) / len(y)
        grad_b = np.mean(y_pred - y)
        w = w - lr * grad_w
        b = b - lr * grad_b

        if step % 20 == 0:
            print(f"step {step}, loss {loss: .4f}")
    return (w, b)

# test
X = np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([0, 0, 0, 1, 1, 1])

w, b = train_logistic_regression(X, y)

print(_sigmoid(X @ w + b))