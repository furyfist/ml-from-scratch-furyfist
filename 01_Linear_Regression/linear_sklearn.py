import numpy as np
import matplotlib.pyplot as plt

# S1 : Generating dummy data
X = np.random.rand(100,1) # means 100 samples, 1 feature
y = 3 * X.squeeze() + 4 + np.random.randn(100) * 0.5 # y = 3x + 4 + noise

# S2 : Initialize Parameters
m = 10.0  # slope
b = 10.0  # intercept
alpha = 0.01  # learning rate
n = 1000  # total iterations
size = X.shape[0]  # number of samples

cost_history = []

# S3 : Gradient descent loop
for i in range(n) :
    y_pred = m * X.squeeze() + b
    error = y_pred - y

    # Compute gradients
    dm = (1 / size) * np.dot(error, X.squeeze())
    db =  (1 / size) * np.sum(error)

    # updated Parameters
    m = m - alpha * dm
    b = b - alpha * db

    # Compute and store loss
    loss = (1 / (2 * size)) * np.sum(error ** 2)
    cost_history.append(loss)

    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")
        

# Step 4: Final Result
print(f"\nFinal model: y = {m:.2f}x + {b:.2f}")

# Step 5 : Plot cost bs Iterations
plt.plot(range(n), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Cost vs Iteration')
plt.grid(True)
plt.show()