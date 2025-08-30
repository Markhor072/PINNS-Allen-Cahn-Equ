# Physics-Informed Neural Networks (PINNs) for the Allen-Cahn Equation

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

An implementation of **Physics-Informed Neural Networks (PINNs)** to solve the non-linear **Allen-Cahn equation**, a foundational model for phase separation and reaction-diffusion systems in material science.

---

## 📖 Theoretical Background

The **Allen-Cahn equation** is a seminal partial differential equation (PDE) in mathematical physics:

$$
\frac{\partial u}{\partial t} - D \frac{\partial^2 u}{\partial x^2} + \gamma (u^3 - u) = 0, \quad x \in [-1, 1], t \in [0, T]
$$

where:
- $u(x, t)$ is the phase field variable,
- $D$ is the diffusion coefficient,
- $\gamma$ is a relaxation parameter.

This project demonstrates how deep learning can be used to solve such complex, non-linear PDEs without traditional numerical discretization methods.

## 🚀 Key Features

- **PINNs Framework**: Implements the core PINN methodology to embed the physics of the Allen-Cahn equation directly into the loss function of a neural network.
- **Custom Loss Function**: The loss combines:
  - **PDE Loss**: Residual of the Allen-Cahn equation.
  - **Boundary Condition (BC) Loss**: Enforces periodic or Dirichlet/Neumann BCs.
  - **Initial Condition (IC) Loss**: Fits the initial state of the system.
- **Spectral Validation**: Validates the neural network solution against a reference solution obtained using a high-accuracy spectral method (when available).

## 🛠️ Tech Stack & Libraries

- **Core Framework**: TensorFlow 2.x / Keras
- **Scientific Computing**: NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Interactive Development**: Jupyter Notebook

## 📁 Repository Structure
PINNS-Allen-Cahn-Equ/
├── Allen_Cahn_PINNs.ipynb # Main Jupyter notebook with full implementation & analysis

├── models/ # (Optional) Directory for saved trained models

├── utils/ # (Optional) Helper functions (e.g., for plotting, data generation)

├── figs/ # Generated plots and visualizations

├── README.md

└── requirements.txt # Python dependencies


## 💡 Methodology

1.  **Problem Setup**: Define the computational domain, parameters (D, γ), and initial/boundary conditions for the Allen-Cahn equation.
2.  **Network Architecture**: Construct a fully connected neural network (NN) $u_{\theta}(x, t)$ to approximate the solution, where $\theta$ represents the network parameters.
3.  **Physics-Informed Loss**: Calculate derivatives of the NN output ($u_t$, $u_{xx}$) using automatic differentiation (TensorFlow's `GradientTape`) and compute the PDE residual.
4.  **Training**: Minimize a composite loss function $L = L_{PDE} + L_{BC} + L_{IC}$ to train the network to satisfy both the data and the underlying physics.
5.  **Analysis & Visualization**: Evaluate the solution and plot the results to show the evolution of the phase field and the convergence of the loss.

## 📈 Results

The PINN successfully learns the dynamics of the Allen-Cahn equation, capturing the key phenomenon of **phase separation** and the motion of interfaces.

- **Solution Accuracy**: The NN solution $u_{\theta}(x, t)$ achieves low PDE residual and fits the initial and boundary conditions.
- **Visualization**: The repository includes plots of the solution over time and space, showing the evolution from the initial condition to the steady state.
- **Loss Convergence**: Plots of the training loss demonstrate the optimization process and the balancing of the different loss components.

## 🔧 Installation & Execution

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Markhor072/PINNS-Allen-Cahn-Equ.git
    cd PINNS-Allen-Cahn-Equ
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    # Or core dependencies:
    # pip install tensorflow numpy matplotlib jupyter
    ```

3.  **Run the Jupyter Notebook**
    ```bash
    jupyter notebook Allen_Cahn_PINNs.ipynb
    ```
    Execute the notebook cells to train the PINN and generate the plots.

## 👨‍💻 Author

**Shahid Hassan**

- GitHub: [@Markhor072](https://github.com/Markhor072)
- LinkedIn: [Shahid Hassan](https://www.linkedin.com/in/markhor072)
- Portfolio: [shahidhassan.vercel.app](https://shahidhassan.vercel.app)

## 📚 References

1.  Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.
2.  Allen, S. M., & Cahn, J. W. (1979). A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening. Acta Metallurgica, 27(6), 1085–1095.
3.  Original PINNs Paper Codebase: [https://github.com/maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
