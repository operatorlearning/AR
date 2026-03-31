import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import interpolate
import time
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Config
# ============================================================
CONFIG = {
    # Data
    "sampling_points": 41,
    "n_samples": 600,
    "length_scale": 0.8,

    # Model
    "trunk_input_dim": 2,
    "trunk_hidden": 512,
    "p": 256 + 128,

    # Training
    "epochs": 1,
    "lambda_": 1e-7,

    # Testing
    "n_test": 10,
    "test_length_scale": 0.8,

    # Visualization
    "colormap": "viridis",
    "dpi": 150,
    "save_path": None,
}
# ============================================================


def vec_to_grid(x, N):
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


def generate(samples=10, begin=0, end=1, random_dim=11, out_dim=101, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    features = np.array([vec_to_grid(y, N=random_dim) for y in features])
    x_grid = np.linspace(begin, end, out_dim)
    return space.eval_u(features, x_grid, x_grid)


def remove_3d_outer(tensor):
    if tensor.dim() != 3:
        raise ValueError(f"Input must be a 3D tensor, current dimension: {tensor.dim()}")
    h, w = tensor.shape[-2], tensor.shape[-1]
    if h < 3 or w < 3:
        raise ValueError(f"Spatial dimension must be at least 3x3, current size: {h}x{w}")
    return tensor[..., 1:-1, 1:-1]


class GRF(object):
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

    def eval_u(self, ys, x, y):
        res = np.zeros((ys.shape[0], x.shape[0], x.shape[0]))
        for i in range(ys.shape[0]):
            res[i] = interpolate.interp2d(self.x, self.x, ys[i], kind=self.interp, copy=False)(x, y)
        return res


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        eps = 0.00001 if y.size()[-1] == 1 else 0
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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        dim = ((39 - 1) - 2) // 2 + 1
        self.classifier = nn.Sequential(
            nn.Linear(dim * dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DeepONetELM2D():
    def __init__(self, trunk_input_dim, trunk_hidden):
        self.branch = CNN().double()
        self.branch_beta = None
        self.trunk_W1 = torch.randn(trunk_input_dim, trunk_hidden * 2).double()
        self.trunk_b1 = torch.randn(1, trunk_hidden * 2).double()
        self.trunk_W2 = torch.randn(trunk_hidden * 2, trunk_hidden).double()
        self.trunk_b2 = torch.randn(1, trunk_hidden).double()
        self.trunk_beta = None

    def _branch_hidden(self, f):
        return self.branch.forward(f)

    def _trunk_hidden(self, x):
        return torch.sigmoid(
            torch.sigmoid(x @ self.trunk_W1 + self.trunk_b1) @ self.trunk_W2 + self.trunk_b2
        )

    def fit(self, data_X, data_f, data_y, epochs=10, lambda_=1e-7, p=2):
        start_time = time.time()  # 开始计时

        n_samples = data_f.shape[0]
        n_points = data_X.shape[0]
        print(f"Training with {n_samples} samples, {n_points} points per sample")

        self.branch_beta = torch.randn(self.trunk_W2.shape[1], p).double()
        self.trunk_beta = torch.randn(self.trunk_W2.shape[1], p).double()

        with torch.no_grad():
            B = self._branch_hidden(data_f).double()
            T = self._trunk_hidden(data_X).double()
        print(f"B shape: {B.shape}, T shape: {T.shape}")

        train_losses = []
        epoch_times = []  # 记录每个epoch的时间

        for epoch in range(epochs):
            epoch_start = time.time()  # epoch开始时间

            # Step 1: fix trunk_beta, update branch_beta
            VT_T = self.trunk_beta.T @ T.T
            BTB_inv = torch.linalg.inv(B.T @ B + lambda_ * torch.eye(B.shape[1]).double())
            VTT_inv = torch.linalg.inv(VT_T @ VT_T.T + lambda_ * torch.eye(p).double())
            self.branch_beta = BTB_inv @ B.T @ data_y @ VT_T.T @ VTT_inv

            # Step 2: fix branch_beta, update trunk_beta
            BT_W = B @ self.branch_beta
            BTW_inv = torch.linalg.inv(BT_W.T @ BT_W + lambda_ * torch.eye(p).double())
            TTT_inv = torch.linalg.inv(T.T @ T + lambda_ * torch.eye(T.shape[1]).double())
            self.trunk_beta = (BTW_inv @ BT_W.T @ data_y @ T @ TTT_inv).T

            with torch.no_grad():
                U_pred = B @ self.branch_beta @ self.trunk_beta.T @ T.T
                loss = torch.mean((U_pred - data_y) ** 2)
                train_losses.append(loss.item())

            epoch_time = time.time() - epoch_start  # epoch耗时
            epoch_times.append(epoch_time)

            print(f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {loss.item():.6e} | Time: {epoch_time:.2f}s")

        total_time = time.time() - start_time  # 总耗时
        avg_epoch_time = np.mean(epoch_times)

        print(f"\n{'=' * 60}")
        print(f"Training completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"{'=' * 60}\n")

        # 绘制训练曲线
        plt.figure(figsize=(10, 5))
        plt.semilogy(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE (log scale)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, which='both', linestyle='--')
        plt.show()

    def predict(self, X, f):
        with torch.no_grad():
            B = self._branch_hidden(f).double()
            T = self._trunk_hidden(X).double()
            return B @ self.branch_beta @ self.trunk_beta.T @ T.T


def FD_AD_2d(f, epsilon=1, meshtype='Shishkin'):
    def p1(x, y): return 0
    def p2(x, y): return 0
    def q(x, y):  return 0

    N = f.shape[-1]
    Nd = f.shape[0]
    sigma = 1 / 2
    grid  = np.linspace(0, 1, N)
    gridS = np.hstack((
        np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1),
        np.linspace(1 - sigma, 1,  int((N - 1) / 2) + 1)[1:]
    ))
    h1 = (1 - sigma) / ((N - 1) / 2)
    h2 = sigma / ((N - 1) / 2)
    fS = np.zeros((Nd, N, N))
    yS = np.zeros((Nd, N, N))
    y  = np.zeros((Nd, N, N))

    for k in range(Nd):
        U = np.zeros(((N - 2) ** 2, (N - 2) ** 2))

        def each_row_of_U(i, hx_1, hx_2, hy_1, hy_2, x, y_):
            U[i, i] = (2*epsilon/hx_1/hx_2 + 2*epsilon/hy_1/hy_2
                       + p1(x, y_)/hx_1 + p2(x, y_)/hy_1 + q(x, y_))
            if i % (N-2) > 0:      U[i, i-1]     = -2*epsilon/hx_1/(hx_1+hx_2) - p1(x, y_)/hx_1
            if i - (N-2) >= 0:     U[i, i-(N-2)] = -2*epsilon/hy_1/(hy_1+hy_2) - p2(x, y_)/hy_1
            if i % (N-2) < N-3:    U[i, i+1]     = -2*epsilon/hx_2/(hx_1+hx_2)
            if i+N-2 < (N-2)**2:   U[i, i+(N-2)] = -2*epsilon/hy_2/(hy_1+hy_2)

        Nm = int((N - 3) / 2)
        for j in range(Nm):
            y_j = (j + 1) * h1
            for i in range(Nm):
                each_row_of_U(j*(N-2)+i, h1, h1, h1, h1, (i+1)*h1, y_j)
            each_row_of_U(j*(N-2)+Nm, h1, h2, h1, h1, (Nm+1)*h1, y_j)
            for i in range(Nm+1, N-2):
                each_row_of_U(j*(N-2)+i, h2, h2, h1, h1, (Nm+1)*h1+(i-Nm)*h2, y_j)

        y_j = (Nm + 1) * h1
        for i in range(Nm):
            each_row_of_U(Nm*(N-2)+i, h1, h1, h1, h2, (i+1)*h1, y_j)
        each_row_of_U(Nm*(N-2)+Nm, h1, h2, h1, h2, (Nm+1)*h1, y_j)
        for i in range(Nm+1, N-2):
            each_row_of_U(Nm*(N-2)+i, h2, h2, h1, h2, (Nm+1)*h1+(i-Nm)*h2, y_j)

        for j in range(Nm+1, N-2):
            y_j = (Nm+1)*h1 + (j-Nm)*h2
            for i in range(Nm):
                each_row_of_U(j*(N-2)+i, h1, h1, h2, h2, (i+1)*h1, y_j)
            each_row_of_U(j*(N-2)+Nm, h1, h2, h2, h2, (Nm+1)*h1, y_j)
            for i in range(Nm+1, N-2):
                each_row_of_U(j*(N-2)+i, h2, h2, h2, h2, (Nm+1)*h1+(i-Nm)*h2, y_j)

        B_vec = np.zeros(((N-2)**2, 1))
        fS[k] = interpolate.interp2d(grid, grid, f[k])(gridS, gridS)
        B_vec[:] = grid_to_vec(fS[k], N-2)
        X = np.linalg.solve(U, B_vec).flatten()
        yS[k] = vec_to_grid(X, N)
        y[k]  = interpolate.interp2d(gridS, gridS, yS[k])(grid, grid)

    return yS if meshtype == 'Shishkin' else y


def visualize_results(X_test, pred, true,
                      figsize=(12, 5), dpi=150, save_path=None, colormap='viridis'):
    y_coords = np.unique(X_test[:, 0])
    x_coords = np.unique(X_test[:, 1])
    X, Y = np.meshgrid(x_coords, y_coords)

    pred_grid = pred.reshape(len(y_coords), len(x_coords))
    true_grid = true.reshape(len(y_coords), len(x_coords))
    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    plt.style.use('seaborn-v0_8-whitegrid')

    # 2D contour
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for ax, data, title in zip(axes, [pred_grid, true_grid], ['Model Prediction', 'Reference Solution']):
        im = ax.contourf(Y, X, data, levels=100, cmap=colormap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

    # Error surface
    error_grid = pred_grid - true_grid
    vmax_err = np.max(np.abs(error_grid))
    Xp, Yp = np.meshgrid(
        np.linspace(0, 1, error_grid.shape[1]),
        np.linspace(0, 1, error_grid.shape[0])
    )

    fig_err = plt.figure(figsize=(8, 6), dpi=dpi, facecolor='white')
    ax_err = fig_err.add_subplot(111, projection='3d')
    norm_err = plt.Normalize(-vmax_err, vmax_err)
    ax_err.plot_surface(Xp, Yp, error_grid, cmap='PiYG', norm=norm_err,
                        linewidth=0.3, edgecolor='gray', alpha=0.85, rcount=30, ccount=30)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.PiYG, norm=norm_err)
    mappable.set_array(error_grid)
    cbar = fig_err.colorbar(mappable, ax=ax_err, shrink=0.5, aspect=15, pad=0.05)
    cbar.set_label('Error', fontsize=10, color='dimgray', labelpad=8)
    cbar.ax.tick_params(labelsize=8, colors='gray')
    cbar.outline.set_edgecolor('lightgray')
    ax_err.set_title('Pointwise Error', fontsize=13, weight='bold', pad=10, color='#2d2d2d')
    ax_err.set_xlabel('$x$', labelpad=8, fontsize=11)
    ax_err.set_ylabel('$y$', labelpad=8, fontsize=11)
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

    # 3D surface
    fig3d = plt.figure(figsize=(13, 5), dpi=dpi, facecolor='white')
    for col, (data, title) in enumerate(zip([pred_grid, true_grid], ['Model Prediction', 'Reference Solution'])):
        ax = fig3d.add_subplot(1, 2, col + 1, projection='3d')
        ax.set_facecolor('white')
        surf = ax.plot_surface(Y, X, data, cmap='coolwarm', vmin=vmin, vmax=vmax,
                               linewidth=0, antialiased=True, alpha=0.95, rcount=80, ccount=80)
        offset = data.min() - 0.15 * (data.max() - data.min())
        ax.contourf(Y, X, data, zdir='z', offset=offset, levels=20, cmap='coolwarm', alpha=0.4)
        ax.set_title(title, fontsize=13, weight='bold', pad=10)
        ax.set_xlabel('$x$', labelpad=6)
        ax.set_ylabel('$y$', labelpad=6)
        ax.set_zlim(offset, data.max())
        ax.view_init(elev=25, azim=-60)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor((0, 0, 0, 0))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid'].update({'color': (0.9, 0.9, 0.9, 1.0), 'linewidth': 0.5})
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
        cbar_ax = fig3d.add_axes([0.47 if col == 0 else 0.955, 0.20, 0.010, 0.55])
        cb = fig3d.colorbar(surf, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=8, colors='gray')
        cb.outline.set_edgecolor('lightgray')
    fig3d.subplots_adjust(left=0.01, right=0.945, top=0.93, bottom=0.07, wspace=0.15)
    if save_path:
        plt.savefig(save_path.replace('.', '_3d.'), bbox_inches='tight', dpi=dpi)
    plt.show()

def show(model, yx_grid, cfg):
    f_test = torch.from_numpy(generate(1, 0, 1, out_dim=cfg["sampling_points"],
                                       length_scale=cfg["test_length_scale"]))
    y_test = remove_3d_outer(torch.from_numpy(FD_AD_2d(f_test)))
    f_test = remove_3d_outer(f_test).double()
    y_test = y_test.reshape(1, -1).double()

    with torch.no_grad():
        pred = model.predict(yx_grid, f_test)
        mse = torch.mean((pred - y_test) ** 2)
        print(f"Test MSE: {mse:.4e}")

    visualize_results(yx_grid.numpy(), pred[0].numpy(), y_test[0].numpy(),
                      dpi=cfg["dpi"], save_path=cfg["save_path"], colormap=cfg["colormap"])


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    cfg = CONFIG
    sp = cfg["sampling_points"]

    # Grid setup
    x = torch.linspace(0, 1, sp)
    y = torch.linspace(0, 1, sp)
    x = x[1:-1]
    y = y[1:-1]
    x_grid, y_grid = torch.meshgrid([x, y], indexing='ij')
    yx_grid = torch.cat([
        x_grid.reshape(-1, 1),
        y_grid.reshape(-1, 1)
    ], dim=1).double()

    # Data generation
    n_samples = cfg["n_samples"]
    f_grf = generate(n_samples, out_dim=sp, length_scale=cfg["length_scale"])
    u_real = FD_AD_2d(f_grf)

    # Data processing
    f_grf = remove_3d_outer(torch.from_numpy(f_grf))
    f_grf = torch.unsqueeze(f_grf, 1).double()
    u_real = remove_3d_outer(torch.from_numpy(u_real))
    u_real = u_real.reshape(n_samples, -1).double()

    print(f"\n=== Solving 2D Poisson Equation with RandONet Form ===")
    print(f"Data shapes:")
    print(f"  yx_grid: {yx_grid.shape}")
    print(f"  f_grf: {f_grf.shape}")
    print(f"  u_real: {u_real.shape}")

    # Model
    model = DeepONetELM2D(
        trunk_input_dim=cfg["trunk_input_dim"],
        trunk_hidden=cfg["trunk_hidden"]
    )

    # Train
    model.fit(yx_grid, f_grf, u_real,
              epochs=cfg["epochs"],
              p=cfg["p"],
              lambda_=cfg["lambda_"])

    # Test
    print("\n=== Testing ===")
    for i in range(cfg["n_test"]):
        print(f"\nTest sample {i + 1}/{cfg['n_test']}:")
        show(model, yx_grid, cfg)