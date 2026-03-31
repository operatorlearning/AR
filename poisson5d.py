import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.interpolate import RegularGridInterpolator


# ==================== 5D Poisson 有限差分求解器 ====================
def FD_5d_poisson(f, epsilon=1):
    """
    求解 5D Poisson 方程: -ε(u_xx + u_yy + u_zz + u_ww + u_vv) = f
    使用 11 点有限差分模板

    参数:
        f: (Nd, N, N, N, N, N) - 右端项
        epsilon: 扩散系数

    返回:
        u: (Nd, N, N, N, N, N) - 数值解
    """
    N = f.shape[-1]
    Nd = f.shape[0]
    h = 1.0 / (N - 1)

    def build_laplacian_5d(N, h, epsilon):
        n = N - 2
        n_total = n ** 5

        print(f"Building sparse Laplacian matrix for {N}^5 grid ({n_total} interior points)...")

        # 使用稀疏矩阵存储
        from scipy.sparse import lil_matrix
        A = lil_matrix((n_total, n_total))

        alpha = epsilon / (h ** 2)

        for v in range(n):  # 5th dimension
            for w in range(n):  # 4th dimension
                for z in range(n):  # 3rd dimension
                    for y in range(n):  # 2nd dimension
                        for x in range(n):  # 1st dimension
                            idx = v * n ** 4 + w * n ** 3 + z * n ** 2 + y * n + x

                            # 中心点
                            A[idx, idx] = 10 * alpha

                            # x 方向
                            if x > 0:
                                A[idx, idx - 1] = -alpha
                            if x < n - 1:
                                A[idx, idx + 1] = -alpha

                            # y 方向
                            if y > 0:
                                A[idx, idx - n] = -alpha
                            if y < n - 1:
                                A[idx, idx + n] = -alpha

                            # z 方向
                            if z > 0:
                                A[idx, idx - n ** 2] = -alpha
                            if z < n - 1:
                                A[idx, idx + n ** 2] = -alpha

                            # w 方向
                            if w > 0:
                                A[idx, idx - n ** 3] = -alpha
                            if w < n - 1:
                                A[idx, idx + n ** 3] = -alpha

                            # v 方向
                            if v > 0:
                                A[idx, idx - n ** 4] = -alpha
                            if v < n - 1:
                                A[idx, idx + n ** 4] = -alpha

        return A.tocsr()

    A = build_laplacian_5d(N, h, epsilon)
    u = np.zeros((Nd, N, N, N, N, N))

    print(f"Solving {Nd} linear systems...")
    from scipy.sparse.linalg import spsolve

    for sample_idx in range(Nd):
        if (sample_idx + 1) % 5 == 0:
            print(f"  Solved {sample_idx + 1}/{Nd} systems")
        f_interior = f[sample_idx, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1].flatten()
        u_interior = spsolve(A, f_interior)
        u[sample_idx, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1] = u_interior.reshape(N - 2, N - 2, N - 2, N - 2, N - 2)

    print("Done solving!")
    return u


# ==================== 5D GRF 生成器 ====================
class GRF5D(object):
    def __init__(self, begin=0, end=1, length_scale=1, N=10):
        self.N = N
        self.begin = begin
        self.end = end
        self.x = np.linspace(begin, end, num=N)

        # 生成所有网格点
        X1, X2, X3, X4, X5 = np.meshgrid(self.x, self.x, self.x, self.x, self.x, indexing='ij')
        self.grid_points = np.stack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel()], axis=1)

        print(f"Computing 5D covariance matrix for {N}^5 = {N ** 5} points...")
        self.K = self._compute_covariance(self.grid_points, length_scale)

        print("Performing Cholesky decomposition...")
        self.L = np.linalg.cholesky(self.K + 1e-10 * np.eye(N ** 5))
        print("GRF5D initialized!")

    def _compute_covariance(self, points, length_scale):
        n = points.shape[0]
        K = np.zeros((n, n))

        batch_size = 1000
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            for j in range(i, n, batch_size):
                end_j = min(j + batch_size, n)

                diff = points[i:end_i, np.newaxis, :] - points[np.newaxis, j:end_j, :]
                dist_sq = np.sum(diff ** 2, axis=2)
                K_block = np.exp(-0.5 * dist_sq / (length_scale ** 2))

                K[i:end_i, j:end_j] = K_block
                if i != j:
                    K[j:end_j, i:end_i] = K_block.T

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Computed {i + batch_size}/{n} rows")

        return K

    def random(self, n_samples, mean=0):
        u = np.random.randn(self.N ** 5, n_samples)
        samples = np.dot(self.L, u).T + mean
        return samples.reshape(n_samples, self.N, self.N, self.N, self.N, self.N)

    def interpolate(self, samples, target_N):
        n_samples = samples.shape[0]
        target_grid = np.linspace(self.begin, self.end, target_N)
        result = np.zeros((n_samples, target_N, target_N, target_N, target_N, target_N))

        print(f"Interpolating {n_samples} samples from {self.N}^5 to {target_N}^5...")
        for i in range(n_samples):
            if (i + 1) % 5 == 0:
                print(f"  Interpolated {i + 1}/{n_samples} samples")
            interpolator = RegularGridInterpolator(
                (self.x, self.x, self.x, self.x, self.x),
                samples[i],
                method='linear'
            )

            X1, X2, X3, X4, X5 = np.meshgrid(target_grid, target_grid, target_grid,
                                             target_grid, target_grid, indexing='ij')
            points = np.stack([X1.ravel(), X2.ravel(), X3.ravel(),
                               X4.ravel(), X5.ravel()], axis=1)
            result[i] = interpolator(points).reshape(target_N, target_N, target_N, target_N, target_N)

        return result


def generate_5d(samples=10, begin=0, end=1, random_dim=5, out_dim=7, length_scale=0.1):
    """生成 5D 高斯随机场"""
    space = GRF5D(begin, end, length_scale=length_scale, N=random_dim)
    features = space.random(samples, mean=0)

    if out_dim != random_dim:
        features = space.interpolate(features, out_dim)

    return features


# ==================== 损失函数 ====================
class LpLoss(object):
    def __init__(self, d=5, p=2, size_average=True, reduction=True):
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


def remove_6d_boundary(tensor):
    """移除 6D 张量的边界（第一维是batch）"""
    if tensor.dim() != 6:
        raise ValueError(f"Input must be a 6D tensor, current dimension: {tensor.dim()}")
    return tensor[:, 1:-1, 1:-1, 1:-1, 1:-1, 1:-1]


# ==================== 5D CNN ====================
class CNN5D(nn.Module):
    def __init__(self, hidden_dim=128):
        super(CNN5D, self).__init__()

        # 使用自定义5D卷积（通过分解为多个3D卷积）
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )

        # 全连接层处理5D特征
        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 1, D1, D2, D3, D4, D5)
        batch_size = x.shape[0]
        D1, D2, D3, D4, D5 = x.shape[2:]

        # 沿着第4和第5维度进行平均池化，降维到3D
        x_3d = x.mean(dim=(4, 5))  # (batch, 1, D1, D2, D3)

        x_3d = self.conv1(x_3d)
        x_3d = self.conv2(x_3d)

        x_flat = torch.flatten(x_3d, 1)
        out = self.fc(x_flat)

        return out


# ==================== DeepONet 5D ====================
class DeepONetELM5D():
    def __init__(self, trunk_input_dim, trunk_hidden):
        super(DeepONetELM5D, self).__init__()
        self.branch = CNN5D(hidden_dim=trunk_hidden).double()
        self.branch_beta = None

        self.trunk_W1 = torch.randn(trunk_input_dim, trunk_hidden * 2).double()
        self.trunk_b1 = torch.randn(1, trunk_hidden * 2).double()
        self.trunk_W2 = torch.randn(trunk_hidden * 2, trunk_hidden).double()
        self.trunk_b2 = torch.randn(1, trunk_hidden).double()
        self.trunk_beta = None

    def _branch_hidden(self, f):
        return self.branch.forward(f)

    def _trunk_hidden(self, x):
        return torch.sigmoid((torch.sigmoid(x @ self.trunk_W1 + self.trunk_b1)) @ self.trunk_W2 + self.trunk_b2)

    def fit(self, data_X, data_f, data_y, epochs=10, lambda_=1e-7, p=2):
        n_samples = data_f.shape[0]
        n_points = data_X.shape[0]

        print(f"\nTraining with {n_samples} samples, {n_points} points per sample")

        self.branch_beta = torch.randn(self.trunk_W2.shape[1], p).double()
        self.trunk_beta = torch.randn(self.trunk_W2.shape[1], p).double()

        print("Computing hidden layers...")
        with torch.no_grad():
            B = self._branch_hidden(data_f).double()
            T = self._trunk_hidden(data_X).double()

        print(f"B shape: {B.shape}, T shape: {T.shape}")
        print(f"data_y shape: {data_y.shape}")

        train_losses = []
        for epoch in range(epochs):
            # 固定 V，求 W
            VT_T = self.trunk_beta.T @ T.T
            BTB = B.T @ B + lambda_ * torch.eye(B.shape[1]).double()
            BTB_inv = torch.linalg.inv(BTB)
            VTT_VTT = VT_T @ VT_T.T + lambda_ * torch.eye(p).double()
            VTT_VTT_inv = torch.linalg.inv(VTT_VTT)
            self.branch_beta = BTB_inv @ B.T @ data_y @ VT_T.T @ VTT_VTT_inv

            # 固定 W，求 V
            BT_W = B @ self.branch_beta
            BTW_BTW = BT_W.T @ BT_W + lambda_ * torch.eye(p).double()
            BTW_BTW_inv = torch.linalg.inv(BTW_BTW)
            TTT = T.T @ T + lambda_ * torch.eye(T.shape[1]).double()
            TTT_inv = torch.linalg.inv(TTT)
            V_T = BTW_BTW_inv @ BT_W.T @ data_y @ T @ TTT_inv
            self.trunk_beta = V_T.T

            with torch.no_grad():
                U_pred = B @ self.branch_beta @ self.trunk_beta.T @ T.T
                train_loss = torch.mean((U_pred - data_y) ** 2)
                train_losses.append(train_loss.item())

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss.item():.6e}")

        plt.figure(figsize=(10, 5))
        plt.semilogy(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE (log scale)')
        plt.title('Training Progress (5D)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--')
        plt.show()

    def predict(self, X, f):
        with torch.no_grad():
            B = self._branch_hidden(f).double()
            T = self._trunk_hidden(X).double()
            U_pred = B @ self.branch_beta @ self.trunk_beta.T @ T.T
            return U_pred


def visualize_5d_results(coords_grid, pred, true, colormap='RdYlBu_r', interpolation='bicubic'):
    """
    5D结果可视化：展示多个3D超平面切片

    固定两个维度，展示剩余3个维度的切面
    """
    n_points = len(pred)
    grid_size = int(round(n_points ** (1 / 5)))

    # 重塑为5D网格
    pred_5d = pred.reshape(grid_size, grid_size, grid_size, grid_size, grid_size)
    true_5d = true.reshape(grid_size, grid_size, grid_size, grid_size, grid_size)

    mid = grid_size // 2

    # 统一颜色范围
    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())

    fig = plt.figure(figsize=(18, 12))

    # 展示6种不同的3D切面（固定2个维度）
    slice_configs = [
        (3, 4, "x-y-z", (0, 1, 2)),  # 固定w,v
        (2, 4, "x-y-w", (0, 1, 3)),  # 固定z,v
        (2, 3, "x-y-v", (0, 1, 4)),  # 固定z,w
        (1, 4, "x-z-w", (0, 2, 3)),  # 固定y,v
        (1, 3, "x-z-v", (0, 2, 4)),  # 固定y,w
        (0, 4, "y-z-w", (1, 2, 3)),  # 固定x,v
    ]

    for idx, (fix_dim1, fix_dim2, title, show_dims) in enumerate(slice_configs):
        # 提取切片
        slices = [slice(None)] * 5
        slices[fix_dim1] = mid
        slices[fix_dim2] = mid

        pred_slice = pred_5d[tuple(slices)]
        true_slice = true_5d[tuple(slices)]

        # 预测值
        ax_pred = plt.subplot(4, 3, idx + 1)
        im = ax_pred.imshow(pred_slice[:, :, mid], cmap=colormap, vmin=vmin, vmax=vmax,
                            origin='lower', aspect='auto', interpolation=interpolation)
        ax_pred.set_title(f'Pred: {title} (mid slice)', fontsize=10, weight='bold')
        plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04)

        # 真实值
        ax_true = plt.subplot(4, 3, idx + 7)
        im = ax_true.imshow(true_slice[:, :, mid], cmap=colormap, vmin=vmin, vmax=vmax,
                            origin='lower', aspect='auto', interpolation=interpolation)
        ax_true.set_title(f'True: {title} (mid slice)', fontsize=10, weight='bold')
        plt.colorbar(im, ax=ax_true, fraction=0.046, pad=0.04)

    plt.tight_layout()

    filename = f'5d_poisson_slices_{colormap}'
    plt.savefig(f'{filename}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}.pdf/png")
    plt.show()


def on_training_data(model, coords_grid_inner, f_train_inner, u_train_inner, test_indices):
    """
    在训练集数据上进行测试

    参数:
        model: 训练好的模型
        coords_grid_inner: 空间坐标网格
        f_train_inner: 训练集输入 (N, 1, D, H, W, U, V)
        u_train_inner: 训练集输出 (N, n_points)
        test_indices: 要测试的样本索引列表
    """
    print("\n" + "=" * 60)
    print("Testing on Training Data")
    print("=" * 60)

    for idx in test_indices:
        print(f"\n{'=' * 40}")
        print(f"Test sample index: {idx}")
        print(f"{'=' * 40}")

        # 提取单个样本
        f_test = f_train_inner[idx:idx + 1]  # (1, 1, D, H, W, U, V)
        y_test = u_train_inner[idx:idx + 1]  # (1, n_points)

        with torch.no_grad():
            pred = model.predict(coords_grid_inner, f_test)
            mse = torch.mean((pred - y_test) ** 2)
            rel_error = torch.norm(pred - y_test) / torch.norm(y_test)
            print(f"Test MSE: {mse:.6e}")
            print(f"Relative L2 Error: {rel_error:.6e}")

        # 可视化
        visualize_5d_results(coords_grid_inner.numpy(), pred[0].numpy(), y_test[0].numpy())


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("5D Poisson Equation Solver with RandONet")
    print("=" * 60)

    # 参数设置（5D需要更小的网格）
    N_train = 100
    grid_size = 7  # 完整网格（包含边界）

    # 生成训练数据
    print(f"\nGenerating {N_train} training samples with grid size {grid_size}...")
    f_train = generate_5d(N_train, out_dim=grid_size, length_scale=1)
    print("\nSolving 5D Poisson equation...")
    u_train = FD_5d_poisson(f_train, epsilon=1)

    # 去掉边界
    print("\nRemoving boundary points...")
    f_train_inner = remove_6d_boundary(torch.from_numpy(f_train))
    u_train_inner = remove_6d_boundary(torch.from_numpy(u_train))

    # 生成内部点的空间网格
    inner_size = grid_size - 2
    coords_inner = torch.linspace(0, 1, grid_size)[1:-1]
    X1, X2, X3, X4, X5 = torch.meshgrid([coords_inner] * 5, indexing='ij')
    coords_grid_inner = torch.stack([X1.ravel(), X2.ravel(), X3.ravel(),
                                     X4.ravel(), X5.ravel()], dim=1).double()

    # 转换格式
    f_train_inner = torch.unsqueeze(f_train_inner, 1).double()
    u_train_inner = u_train_inner.reshape(N_train, -1).double()

    print(f"\nData shapes:")
    print(f"  coords_grid_inner: {coords_grid_inner.shape}")
    print(f"  f_train_inner: {f_train_inner.shape}")
    print(f"  u_train_inner: {u_train_inner.shape}")

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    # 初始化模型
    print("\nInitializing model...")
    model = DeepONetELM5D(
        trunk_input_dim=5,
        trunk_hidden=512
    )

    # 训练
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    model.fit(coords_grid_inner, f_train_inner, u_train_inner, epochs=1, p=512, lambda_=1e-8)

    # 测试：使用训练集中的样本
    test_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # 选择10个训练样本进行测试
    on_training_data(model, coords_grid_inner, f_train_inner, u_train_inner, test_indices)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)