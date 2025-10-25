
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import MeanAbsoluteError
from pathlib import Path

# ========== CẤU HÌNH ==========
ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = ROOT / "data/processed_data.npz"
CKPT_DIR = ROOT / "models"
CKPT_PATH = CKPT_DIR / "best_lstm_manual.pt"

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10  # early stopping

class StormSeqDataset(Dataset):
    def __init__(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X = torch.tensor(X, dtype=torch.float32)
        if y.ndim >= 3 and y.shape[1] == 1:
            y = np.squeeze(y, axis=1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class _ManualLSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        # U*: input->hidden (có bias)
        self.Uf = nn.Linear(in_dim, hidden, bias=True)
        self.Ui = nn.Linear(in_dim, hidden, bias=True)
        self.Uo = nn.Linear(in_dim, hidden, bias=True)
        self.Ug = nn.Linear(in_dim, hidden, bias=True)
        # W*: hidden->hidden (không bias)
        self.Wf = nn.Linear(hidden, hidden, bias=False)
        self.Wi = nn.Linear(hidden, hidden, bias=False)
        self.Wo = nn.Linear(hidden, hidden, bias=False)
        self.Wg = nn.Linear(hidden, hidden, bias=False)

        # Khởi tạo
        for lin in [self.Uf, self.Ui, self.Uo, self.Ug]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in [self.Wf, self.Wi, self.Wo, self.Wg]:
            nn.init.orthogonal_(lin.weight)
        # Forget bias dương để khuyến khích "nhớ" lúc đầu
        with torch.no_grad():
            self.Uf.bias.fill_(1.0)

    def forward(self, x_t, h_prev, c_prev):
        f_t = torch.sigmoid(self.Uf(x_t) + self.Wf(h_prev))
        i_t = torch.sigmoid(self.Ui(x_t) + self.Wi(h_prev))
        o_t = torch.sigmoid(self.Uo(x_t) + self.Wo(h_prev))
        g_t = torch.tanh(   self.Ug(x_t) + self.Wg(h_prev))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMFromScratchForecaster(nn.Module):
    """
    Stacked LSTM từ _ManualLSTMCell. Lấy h ở timestep cuối -> Linear head ra out_dim.
    """
    def __init__(self, in_dim, hidden=20, num_layers=2, out_dim=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout_p = dropout if num_layers > 1 else 0.0

        self.cells = nn.ModuleList([
            _ManualLSTMCell(in_dim if l == 0 else hidden, hidden)
            for l in range(num_layers)
        ])
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.head = nn.Linear(hidden, out_dim)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, T, F]
        B, T, _ = x.size()
        device = x.device
        dtype = x.dtype

        hs = [torch.zeros(B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]
        cs = [torch.zeros(B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h_l, c_l = cell(inp, hs[l], cs[l])
                hs[l], cs[l] = h_l, c_l
                # dropout giữa các layer (không áp cho layer cuối)
                inp = self.dropout(h_l) if (l < self.num_layers - 1) else h_l

        last_h = hs[-1]          # [B, H] tại timestep cuối
        return self.head(last_h) # [B, out_dim]


@torch.no_grad()
def _eval_epoch(loader, model, crit, device, mae_metric):
    model.eval()
    total_loss, n_samples = 0.0, 0
    mae_metric.reset()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        mae_metric.update(pred, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / max(n_samples, 1), mae_metric.compute()

def _train_epoch(loader, model, crit, opt, device, mae_metric):
    model.train()
    total_loss, n_samples = 0.0, 0
    mae_metric.reset()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
        opt.step()

        mae_metric.update(pred, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / max(n_samples, 1), mae_metric.compute()

def run_epoch(loader, model, crit, opt, device, train_mode, mae_metric):
    if train_mode:
        return _train_epoch(loader, model, crit, opt, device, mae_metric)
    else:
        return _eval_epoch(loader, model, crit, device, mae_metric)

def main():
    if not DATA_NPZ.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_NPZ}")

    # 1) Load dữ liệu
    npz = np.load(DATA_NPZ, allow_pickle=True)
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_valid, y_valid = npz["X_valid"], npz["y_valid"]
    X_test,  y_test  = npz["X_test"],  npz["y_test"]
    INPUT_FEATURES = list(npz["INPUT_FEATURES"]) if "INPUT_FEATURES" in npz.files else None
    TARGET_FEATURES = list(npz["TARGET_FEATURES"]) if "TARGET_FEATURES" in npz.files else None

    # 2) out_dim 
    out_dim = len(TARGET_FEATURES) if TARGET_FEATURES else (y_train.shape[-1] if y_train.ndim >= 2 else 1)
    in_dim = X_train.shape[2]

    # 3) DataLoader
    train_loader = DataLoader(StormSeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(StormSeqDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(StormSeqDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    # 4) Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 5) Model + tối ưu + metric 
    model = LSTMFromScratchForecaster(in_dim=in_dim, hidden=20, num_layers=2, out_dim=out_dim, dropout=0.2).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    mae_metric = MeanAbsoluteError().to(device)

    # 6) Checkpoint dir
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # 7) Train + Early stopping 
    best_loss, bad_epochs = float("inf"), 0
    print("\nBắt đầu huấn luyện...")
    for ep in range(1, EPOCHS + 1):
        tr_loss, _ = run_epoch(train_loader, model, crit, opt, device, True, mae_metric)
        va_loss, va_mae = run_epoch(valid_loader, model, crit, None, device, False, mae_metric)

        sched.step(va_loss)

        print(f"Epoch {ep:02d} | Train Loss {tr_loss:.6f} | Valid Loss {va_loss:.6f} | Valid MAE {va_mae:.6f}")

        if va_loss < best_loss - 1e-12:
            best_loss, bad_epochs = va_loss, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_features": INPUT_FEATURES,
                    "target_features": TARGET_FEATURES,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "config": {"hidden": 20, "num_layers": 2, "dropout": 0.2},
                },
                CKPT_PATH,
            )
            print(" -> Đã lưu model tốt nhất.")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping do không cải thiện trên tập validation.")
                break

    # 8) Test
    print("\nĐánh giá trên tập test...")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mae = run_epoch(test_loader, model, crit, None, device, False, mae_metric)
    print(f"[KẾT QUẢ TEST] MSE: {test_loss:.6f} | MAE: {test_mae:.6f}")

# chạy
main()
