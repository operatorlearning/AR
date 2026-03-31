import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Uniform
from scipy import interpolate
import os


# ============= Configuration Class =============
class Config:
    """Unified hyperparameter management"""
    # Physical parameters
    L = 1.0  # Spatial length
    T = 1.0  # Time length
    nu = 0.01  # Viscosity coefficient

    # Grid parameters
    Nt = 80  # Time steps
    Nx = 40  # Spatial steps

    # GRF parameters
    random_dim = 11
    length_scale = 0.3
    interp = "cubic"

    # CNN parameters
    input_height = 78  # Nt - 2
    input_width = 38  # Nx - 2
    cnn_hidden_dim1 = 128
    cnn_hidden_dim2 = 256

    # Model parameters
    trunk_input_dim = 2
    trunk_hidden = 512

    # Training parameters
    n_samples = 1000
    epochs = 2
    rank_p = 384  # 256 + 128
    lambda_reg = 1e-7

    # Testing parameters
    n_test_samples = 10

    # Visualization parameters
    figsize = (12, 5)
    dpi = 150
    colormap = 'viridis'


# ============= Original Code (unchanged) =============
# Parameter settings
L = Config.L
T = Config.T


def FD_burgers_linear(f, Nd, Nt, Nx, nu=Config.nu):
    """Heat equation solver: u_t = nu*u_xx + f"""
    dx = L / Nx
    dt = T / Nt
    u = np.zeros((Nd, Nt + 1, Nx + 1))
    for k in range(Nd):
        for n in range(Nt):
            un = u[k, n]
            u_new = np.zeros(Nx + 1)
            for j in range(1, Nx):
                diff = nu * (un[j + 1] - 2 * un[j] + un[j - 1]) / (dx ** 2)
                u_new[j] = un[j] + dt * diff + dt * f[k, n, j]
            u_new[0] = u_new[-1] = 0
            u[k, n + 1] = u_new
    return u


def vec_to_grid(x, N):
    """Convert vector to grid"""
    res = np.zeros((N, N))
    if N ** 2 == x.shape[0]:
        for i in range(N):
            res[i] = x[i * N:(i + 1) * N].T
    elif N ** 2 > x.shape[0]:
        for i in range(1, N - 1):
            res[i, 1:-1] = x[(i - 1) * (N - 2):i * (N - 2)].T
    else:
        for i in range(N):
            res[i] = x[(i + 1) * (N + 2) + 1:(i + 2) * (N + 2) - 1].T
    return res


def grid_to_vec(X, N):
    """Convert grid to vector"""
    n = X.shape[0]
    res = np.zeros((N ** 2, 1))
    if n == N:
        for i in range(N):
            res[i * N:(i + 1) * N] = X[i][:, None]
    elif n == N + 2:
        for i in range(N):
            res[i * N:(i + 1) * N] = X[i + 1, 1:-1][:, None]
    elif N == n + 2:
        for i in range(1, N - 1):
            res[i * N + 1:(i + 1) * N - 1] = X[i - 1][:, None]
    return res


def generate(samples=10, begin=0, end=1, random_dim=Config.random_dim,
             Nt=Config.Nt, Nx=Config.Nx, length_scale=Config.length_scale,
             interp=Config.interp, A=0):
    """Generate random functions using GRF"""
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    features = np.array([vec_to_grid(y, N=random_dim) for y in features])
    t_grid = np.linspace(begin, end, Nt)
    x_grid = np.linspace(begin, end, Nx)
    x_data = space.eval_u(features, t_grid, x_grid)
    return x_data


class GRF(object):
    """Gaussian Random Field generator"""

    def __init__(self, begin=0, end=1, length_scale=5, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)
        self.z = np.zeros((self.N ** 2, 2))
        for j in range(self.N):
            for i in range(self.N):
                self.z[j * self.N + i] = [self.x[i], self.x[j]]
        self.K = np.exp(-0.5 * self.distance_matrix(self.z, length_scale))
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N ** 2))

    def distance_matrix(self, x, length_scale):
        n = x.shape[0]
        grid = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                grid[i][j] = ((x[i][0] - x[j][0]) ** 2 + (x[i][1] - x[j][1]) ** 2) / length_scale ** 2
                grid[j][i] = grid[i][j]
        return grid

    def random(self, n, A):
        u = np.random.randn(self.N ** 2, n)
        return np.dot(self.L, u).T + A

    def eval_u(self, ys, t, x):
        res = np.zeros((ys.shape[0], t.shape[0], x.shape[0]))
        for i in range(ys.shape[0]):
            f_interp = interpolate.RectBivariateSpline(self.x, self.x, ys[i], kx=3, ky=3)
            res[i] = f_interp(t, x)
        return res


def remove_2d_outer(tensor):
    """Remove the outermost border of each 2D slice in a 3D tensor"""
    if tensor.dim() != 3:
        raise ValueError(f"Input must be a 3D tensor, current dimension: {tensor.dim()}")
    h, w = tensor.shape[-2], tensor.shape[-1]
    if h < 3 or w < 3:
        raise ValueError(f"Spatial dimension must be at least 3x3, current size: {h}x{w}")
    return tensor[..., 1:-1, 1:-1]


class LpLoss(object):
    """Lp loss function"""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        if y.size()[-1] == 1:
            eps = 0.00001
        else:
            eps = 0
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + eps))
            else:
                return torch.sum(diff_norms / (y_norms + eps))
        return diff_norms / (y_norms + eps)

    def __call__(self, x, y):
        return self.rel(x, y)


def weighted_mse_loss(y_pred, y_true, weights):
    """Custom weighted MSE loss function"""
    squared_errors = torch.square(y_pred - y_true)
    weighted_errors = squared_errors * weights
    return torch.sum(weighted_errors)


class CNN(nn.Module):
    """Branch network using CNN"""

    def __init__(self, input_height=Config.input_height, input_width=Config.input_width):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # Calculate output dimension after conv and pooling
        # After conv: (H-1, W-1) = (37, 17)
        # After pooling: (37//2, 17//2) = (18, 8)
        dim_h = ((input_height - 1) - 2) // 2 + 1
        dim_w = ((input_width - 1) - 2) // 2 + 1

        self.classifier = nn.Sequential(
            nn.Linear(dim_h * dim_w, Config.cnn_hidden_dim1),
            nn.Sigmoid(),
            nn.Linear(Config.cnn_hidden_dim1, Config.cnn_hidden_dim2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Create uniform distribution instance
uniform_dist = Uniform(low=-1.0, high=1.0)

class Alternate_RandONet():
    """DeepONet with ELM using RandONet matrix form"""

    def __init__(self, trunk_input_dim=Config.trunk_input_dim, trunk_hidden=Config.trunk_hidden):
        super(Alternate_RandONet, self).__init__()
        self.branch = CNN().double()
        self.branch_beta = None

        self.trunk_W1 = torch.randn(trunk_input_dim, trunk_hidden * 2).double()
        self.trunk_b1 = torch.randn(1, trunk_hidden * 2).double()
        self.trunk_W2 = torch.randn(trunk_hidden * 2, trunk_hidden).double()
        self.trunk_b2 = torch.randn(1, trunk_hidden).double()
        self.trunk_beta = None

    def _branch_hidden(self, f):
        """Compute branch network hidden layer output"""
        return self.branch.forward(f)

    def _trunk_hidden(self, x):
        """Compute trunk network hidden layer output"""
        return torch.sigmoid((torch.sigmoid(x @ self.trunk_W1 + self.trunk_b1)) @ self.trunk_W2 + self.trunk_b2)

    def fit(self, data_X, data_f, data_y, epochs=Config.epochs, lambda_=Config.lambda_reg, p=Config.rank_p):
        """
        Train using RandONet matrix form

        Args:
            data_X: (n_points, 2) - spatiotemporal coordinates (t, x)
            data_f: (n_samples, 1, Nt, Nx) - input function
            data_y: (n_samples, n_points) - target output matrix
            epochs: number of training epochs
            lambda_: regularization parameter
            p: rank parameter for low-rank approximation
        """
        n_samples = data_f.shape[0]
        n_points = data_X.shape[0]

        print(f"Training with {n_samples} samples, {n_points} points per sample")

        # Initialize parameters
        self.branch_beta = torch.randn(self.trunk_W2.shape[1], p).double()
        self.trunk_beta = torch.randn(self.trunk_W2.shape[1], p).double()

        # Compute hidden layers (only once)
        with torch.no_grad():
            B = self._branch_hidden(data_f).double()  # (n_samples, Lb)
            T = self._trunk_hidden(data_X).double()  # (n_points, Lt)

        print(f"B shape: {B.shape}, T shape: {T.shape}")

        # Alternating optimization
        train_losses = []
        for epoch in range(epochs):
            # === Step 1: Fix V (trunk_beta), solve W (branch_beta) ===
            VT_T = self.trunk_beta.T @ T.T  # (p, n_points)

            BTB = B.T @ B + lambda_ * torch.eye(B.shape[1]).double()
            BTB_inv = torch.linalg.inv(BTB)

            VTT_VTT = VT_T @ VT_T.T + lambda_ * torch.eye(p).double()
            VTT_VTT_inv = torch.linalg.inv(VTT_VTT)

            self.branch_beta = BTB_inv @ B.T @ data_y @ VT_T.T @ VTT_VTT_inv

            # === Step 2: Fix W (branch_beta), solve V (trunk_beta) ===
            BT_W = B @ self.branch_beta  # (n_samples, p)

            BTW_BTW = BT_W.T @ BT_W + lambda_ * torch.eye(p).double()
            BTW_BTW_inv = torch.linalg.inv(BTW_BTW)

            TTT = T.T @ T + lambda_ * torch.eye(T.shape[1]).double()
            TTT_inv = torch.linalg.inv(TTT)

            V_T = BTW_BTW_inv @ BT_W.T @ data_y @ T @ TTT_inv
            self.trunk_beta = V_T.T

            # === Compute loss ===
            with torch.no_grad():
                U_pred = B @ self.branch_beta @ self.trunk_beta.T @ T.T
                train_loss = torch.mean((U_pred - data_y) ** 2)
                train_losses.append(train_loss.item())

                print(f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss.item():.6e}")

    def predict(self, X, f):
        """
        Predict function

        Args:
            X: (n_points, 2) - spatiotemporal coordinates
            f: (n_samples, 1, Nt, Nx) - input function

        Returns:
            (n_samples, n_points) - prediction matrix
        """
        with torch.no_grad():
            B = self._branch_hidden(f).double()
            T = self._trunk_hidden(X).double()
            U_pred = B @ self.branch_beta @ self.trunk_beta.T @ T.T
            return U_pred


def visualize_results(X_test, pred, true,
                      figsize=Config.figsize,
                      dpi=Config.dpi,
                      save_path=None,
                      colormap=Config.colormap):
    """Visualize prediction results with 2D heatmaps, error plot, and 3D surfaces"""
    t_coords = np.unique(X_test[:, 0])
    x_coords = np.unique(X_test[:, 1])
    T_grid, X_grid = np.meshgrid(t_coords, x_coords, indexing='ij')

    pred_grid = pred.reshape(len(t_coords), len(x_coords))
    true_grid = true.reshape(len(t_coords), len(x_coords))

    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    plt.style.use('seaborn-v0_8-whitegrid')

    # ==== 2D Heatmap Comparison ====
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    im1 = axes[0].contourf(X_grid, T_grid, pred_grid, levels=100, cmap=colormap, vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title('Model Prediction', fontsize=14, weight='bold')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$t$')

    im2 = axes[1].contourf(X_grid, T_grid, true_grid, levels=100, cmap=colormap, vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title('Reference Solution', fontsize=14, weight='bold')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$t$')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

    # ==== Error Plot ====
    error_grid = pred_grid - true_grid
    vmax_err = np.max(np.abs(error_grid))

    fig_err = plt.figure(figsize=(8, 6), dpi=dpi, facecolor='white')
    ax_err = fig_err.add_subplot(111, projection='3d')

    norm_err = plt.Normalize(-vmax_err, vmax_err)
    surf_err = ax_err.plot_surface(
        X_grid, T_grid, error_grid,
        cmap='PiYG',
        norm=norm_err,
        linewidth=0.3,
        edgecolor='gray',
        alpha=0.85,
        rcount=30, ccount=30)

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.PiYG, norm=norm_err)
    mappable.set_array(error_grid)
    cbar = fig_err.colorbar(mappable, ax=ax_err, shrink=0.5, aspect=15, pad=0.05)
    cbar.set_label('Error', fontsize=10, color='dimgray', labelpad=8)
    cbar.ax.tick_params(labelsize=8, colors='gray')
    cbar.outline.set_edgecolor('lightgray')

    ax_err.set_title('Pointwise Error', fontsize=13, weight='bold', pad=10, color='#2d2d2d')
    ax_err.set_xlabel('$x$', labelpad=8, fontsize=11)
    ax_err.set_ylabel('$t$', labelpad=8, fontsize=11)
    ax_err.set_zlabel('Error', labelpad=8, fontsize=11)
    ax_err.set_zlim(-vmax_err * 1.2, vmax_err * 1.2)
    ax_err.view_init(elev=25, azim=-60)
    ax_err.set_box_aspect([1, 1, 0.6])

    for pane in [ax_err.xaxis.pane, ax_err.yaxis.pane, ax_err.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        pane.set_edgecolor('black')

    ax_err.xaxis._axinfo['grid'].update({'color': (0.7, 0.7, 0.7, 0.6), 'linewidth': 0.4})
    ax_err.yaxis._axinfo['grid'].update({'color': (0.7, 0.7, 0.7, 0.6), 'linewidth': 0.4})
    ax_err.zaxis._axinfo['grid']['linewidth'] = 0

    ax_err.set_zticklabels([])
    ax_err.set_zticks([])

    ax_err.tick_params(axis='both', colors='black', labelsize=8, length=3, width=0.5)
    ax_err.xaxis.label.set_color('black')
    ax_err.yaxis.label.set_color('black')
    ax_err.zaxis.label.set_color('black')
    ax_err.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_err.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.tight_layout()
    plt.show()

    # ==== 3D Surface Plot ====
    fig3d = plt.figure(figsize=(13, 5), dpi=dpi, facecolor='white')

    for col, (data, title) in enumerate(zip([pred_grid, true_grid], ['Model Prediction', 'Reference Solution'])):
        ax = fig3d.add_subplot(1, 2, col + 1, projection='3d')
        ax.set_facecolor('white')

        surf = ax.plot_surface(
            X_grid, T_grid, data,
            cmap='coolwarm',
            vmin=vmin, vmax=vmax,
            linewidth=0,
            antialiased=True,
            alpha=0.95,
            rcount=80, ccount=80
        )

        offset = data.min() - 0.15 * (data.max() - data.min())
        ax.contourf(X_grid, T_grid, data, zdir='z', offset=offset, levels=20, cmap='coolwarm', alpha=0.4)

        ax.set_title(title, fontsize=13, weight='bold', pad=10)
        ax.set_xlabel('$x$', labelpad=6)
        ax.set_ylabel('$t$', labelpad=6)
        ax.set_zlim(offset, data.max())
        ax.view_init(elev=25, azim=-60)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor((0, 0, 0, 0))

        ax.xaxis._axinfo['grid'].update({'color': (0.9, 0.9, 0.9, 1.0), 'linewidth': 0.5})
        ax.yaxis._axinfo['grid'].update({'color': (0.9, 0.9, 0.9, 1.0), 'linewidth': 0.5})
        ax.zaxis._axinfo['grid'].update({'color': (0.9, 0.9, 0.9, 1.0), 'linewidth': 0.5})

        ax.zaxis.line.set_color((0, 0, 0, 0))
        ax.zaxis.label.set_visible(False)
        ax.zaxis.pane.fill = False
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.set_zticks([])

        ax.tick_params(axis='both', colors='gray', labelsize=9, length=3, width=0.5)
        ax.xaxis.label.set_color('dimgray')
        ax.yaxis.label.set_color('dimgray')
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        cbar_left = 0.47 if col == 0 else 0.955
        cbar_ax = fig3d.add_axes([cbar_left, 0.20, 0.010, 0.55])
        cb = fig3d.colorbar(surf, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8, colors='gray')
        cb.outline.set_edgecolor('lightgray')

    fig3d.subplots_adjust(left=0.01, right=0.945, top=0.93, bottom=0.07, wspace=0.15)

    if save_path:
        plt.savefig(save_path.replace('.', '_3d.'), bbox_inches='tight', dpi=dpi)
    plt.show()


def show(model, tx_grid):
    """Test and visualize a single sample"""
    f_test = torch.from_numpy(generate(1, 0, 1, Nt=Config.Nt, Nx=Config.Nx, length_scale=Config.length_scale))
    y_test = remove_2d_outer(torch.from_numpy(FD_burgers_linear(f_test, Nd=1, Nt=Config.Nt - 1, Nx=Config.Nx - 1)))
    f_test = remove_2d_outer(f_test).double()
    f_test = torch.unsqueeze(f_test, 1)
    y_test = y_test.reshape(1, -1).double()

    with torch.no_grad():
        pred = model.predict(tx_grid, f_test)
        mse = torch.mean((pred - y_test) ** 2)
        print(f"Test MSE: {mse:.4e}")

    visualize_results(tx_grid.numpy(), pred[0].numpy(), y_test[0].numpy())


if __name__ == "__main__":
    # Generate spatiotemporal grid (t, x)
    t = torch.linspace(0, 1, Config.Nt)
    x = torch.linspace(0, 1, Config.Nx)
    t = t[1:-1]
    x = x[1:-1]
    t_grid, x_grid = torch.meshgrid([t, x], indexing='ij')
    t_grid = t_grid.reshape(-1, 1)
    x_grid = x_grid.reshape(-1, 1)
    tx_grid = torch.cat([t_grid, x_grid], dim=1).double()

    # Generate training data
    f_grf = generate(Config.n_samples, Nt=Config.Nt, Nx=Config.Nx, length_scale=Config.length_scale)
    u_real = FD_burgers_linear(f_grf, Nd=Config.n_samples, Nt=Config.Nt - 1, Nx=Config.Nx - 1)

    # Process data
    f_grf = remove_2d_outer(torch.from_numpy(f_grf))
    f_grf = torch.unsqueeze(f_grf, 1).double()
    u_real = remove_2d_outer(torch.from_numpy(u_real))
    u_real = u_real.reshape(Config.n_samples, -1).double()

    print(f"Data shapes:")
    print(f"  tx_grid: {tx_grid.shape}")
    print(f"  f_grf: {f_grf.shape}")
    print(f"  u_real: {u_real.shape}")

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    print(f"\n=== Solving 2D Heat Equation with RandONet Form ===")

    # Initialize model
    model = Alternate_RandONet(
        trunk_input_dim=Config.trunk_input_dim,
        trunk_hidden=Config.trunk_hidden
    )

    # Training
    model.fit(tx_grid, f_grf, u_real, epochs=Config.epochs, p=Config.rank_p, lambda_=Config.lambda_reg)

    # Testing
    print("\n=== Testing ===")
    for i in range(Config.n_test_samples):
        print(f"\nTest sample {i + 1}/{Config.n_test_samples}:")
        show(model, tx_grid)