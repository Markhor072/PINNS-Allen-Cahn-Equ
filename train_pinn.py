import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 512)  # Increased neurons
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)  # Output: u(t, x)

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)  # Concatenate t and x
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return self.fc6(x)

# Define the exact solution for Allen-Cahn equation
def exact_solution(t, x, epsilon=0.01):
    return torch.exp(-epsilon * np.pi**2 * t) * torch.sin(np.pi * x)

# Define the loss function for Allen-Cahn equation
def loss_function(model, t, x, epsilon=0.01):
    # Enable gradient computation
    t.requires_grad = True
    x.requires_grad = True

    # Predict u(t, x)
    u = model(t, x)

    # Compute derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Allen-Cahn residual: u_t - epsilon^2 * u_xx + u^3 - u
    pde_residual = u_t - epsilon**2 * u_xx + u**3 - u

    # Initial condition loss: u(0, x) = sin(pi * x)
    u_initial = torch.sin(np.pi * x)  # Example initial condition
    initial_loss = torch.mean((model(torch.zeros_like(x), x) - u_initial) ** 2)

    # Boundary condition loss: u(t, 0) = 0 and u(t, 1) = 0
    boundary_loss = torch.mean((model(t, torch.zeros_like(t))) ** 2 + (model(t, torch.ones_like(t))) ** 2)

    # Weighting factors for loss terms
    lambda_pde = 1.0
    lambda_ic = 1000.0  # Higher weight for initial condition
    lambda_bc = 1000.0  # Higher weight for boundary conditions

    # Total loss
    loss = lambda_pde * torch.mean(pde_residual ** 2) + lambda_ic * initial_loss + lambda_bc * boundary_loss
    return loss

# Define the training data
t_train = torch.linspace(0, 1, 200).reshape(-1, 1).double()  # More time steps
x_train = torch.linspace(0, 1, 200).reshape(-1, 1).double()  # More spatial points

# Initialize the model, optimizer, and scheduler
model = PINN().double()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjusted learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)

# Training loop
for epoch in range(20000):  # Increased number of epochs
    optimizer.zero_grad()

    # Compute loss
    loss = loss_function(model, t_train, x_train)

    # Backward pass and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()
    scheduler.step(loss)  # Update learning rate based on loss

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model at specific time steps
t_eval = torch.tensor([0.10, 0.90]).reshape(-1, 1).double()  # Time steps
x_eval = torch.linspace(0, 1, 100).reshape(-1, 1).double()  # Spatial points

# Create a grid of (t, x) points
t_grid, x_grid = torch.meshgrid(t_eval.squeeze(), x_eval.squeeze(), indexing="ij")
t_grid = t_grid.reshape(-1, 1)  # Flatten to (num_points, 1)
x_grid = x_grid.reshape(-1, 1)  # Flatten to (num_points, 1)

# Predict u(t, x) using the trained model
u_pred = model(t_grid, x_grid).detach()

# Reshape the predictions to match the grid shape
u_pred = u_pred.reshape(len(t_eval), len(x_eval))  # Shape: (num_time_steps, num_spatial_points)

# Compute the exact solution
u_exact = exact_solution(t_grid, x_grid).reshape(len(t_eval), len(x_eval))

# Compute relative L2 error
relative_error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
print(f"Relative L2 Error: {relative_error.item()}")

# Plot the results
plt.figure(figsize=(10, 6))
for i, t in enumerate(t_eval):
    plt.plot(x_eval.numpy(), u_exact[i].numpy(), label=f"Exact Solution (t = {t.item()})", linestyle="--", linewidth=2)
    plt.plot(x_eval.numpy(), u_pred[i].numpy(), label=f"PINN Solution (t = {t.item()})", linestyle="-", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(t, x)")
plt.title("Comparison of PINN and Exact Solution for Allen-Cahn Equation")
plt.legend()
plt.grid(True)
plt.savefig("pinn_allen_cahn.png")  # Save the plot as a PNG file
plt.show()