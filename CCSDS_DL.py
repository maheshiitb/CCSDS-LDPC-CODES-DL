# gpu_ready_ldpc_training.py
# ----------------------------
# Imports & device
# ----------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# GF(2) linear algebra utilities (kept in NumPy; small one-time cost)
# ----------------------------
def gf2_row_echelon(A):
    A = A.copy() % 2
    nrows, ncols = A.shape
    r = 0
    pivots = []
    for c in range(ncols):
        if r >= nrows:
            break
        pivot = None
        for i in range(r, nrows):
            if A[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(nrows):
            if i != r and A[i, c] == 1:
                A[i] ^= A[r]
        pivots.append(c)
        r += 1
    return A, pivots

def gf2_rank(A):
    _, pivots = gf2_row_echelon(A)
    return len(pivots)

def gf2_inv(A):
    A = A.copy() % 2
    n = A.shape[0]
    I = np.eye(n, dtype=np.uint8)
    aug = np.concatenate((A, I), axis=1).astype(np.uint8)
    r = 0
    for c in range(n):
        if r >= n:
            break
        pivot = None
        for i in range(r, n):
            if aug[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]
        for i in range(n):
            if i != r and aug[i, c] == 1:
                aug[i] ^= aug[r]
        r += 1
    left = aug[:, :n]
    if not np.array_equal(left % 2, np.eye(n, dtype=np.uint8)):
        raise ValueError("Matrix is singular over GF(2); cannot invert.")
    return aug[:, n:] % 2

def gf2_nullspace(A):
    A = A.copy() % 2
    k, n = A.shape
    R, pivots = gf2_row_echelon(A)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    basis = []
    for f in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[f] = 1
        for row_idx, p in enumerate(pivots):
            if R[row_idx, f] == 1:
                x[p] = 1
        basis.append(x)
    if basis:
        H = np.stack(basis, axis=0) % 2
    else:
        H = np.zeros((0, n), dtype=np.uint8)
    return H

# ----------------------------
# Generator matrix (uses GF(2) utilities) - small one-time CPU computation
# ----------------------------
def gm_generation():
    M = 16 
    B = np.eye(M, dtype=np.uint8)
    bZ = np.zeros((M, M), dtype=np.uint8)
    def r(I, s):
        return np.roll(I, s, axis=1)
    Ha = np.concatenate((
        (B ^ r(B, 7)),
        r(B, 2),
        r(B, 14),
        r(B, 6),
        bZ,
        B,
        r(B, 13),
        B
    ), axis=1)

    Hb = np.concatenate((
        r(B, 6),
        (B ^ r(B, 15)),
        B,
        r(B, 1),
        B,
        bZ,
        B,
        r(B, 7)
    ), axis=1)

    Hc = np.concatenate((
        r(B, 4),
        r(B, 1),
        (B ^ r(B, 15)),
        r(B, 14),
        r(B, 11),
        B,
        bZ,
        r(B, 3)
    ), axis=1)

    Hd = np.concatenate((
        B,
        r(B, 1),
        r(B, 9),
        (B ^ r(B, 13)),
        r(B, 14),
        r(B, 1),
        B,
        bZ
    ), axis=1)
    Hm = np.concatenate((Ha, Hb, Hc, Hd), axis=0).astype(np.uint8)
    assert Hm.shape == (4*M, 8*M)
    nrows, ncols = Hm.shape
    left = Hm[:, 0:(ncols//2)]
    right = Hm[:, (ncols//2):]
    right_inv = gf2_inv(right)
    P = (right_inv.dot(left) % 2).astype(np.uint8)
    k = 4 * M
    I_k = np.eye(k, dtype=np.uint8)
    Gm = np.concatenate((I_k, P.T % 2), axis=1).astype(np.uint8)
    return Gm, Hm

# ----------------------------
# Prepare matrices and move to GPU where applicable
# ----------------------------
np.random.seed(0)
Gm_np, Hm_np = gm_generation()   # small CPU step
# Convert to torch tensors on device
Gm = torch.from_numpy(Gm_np.astype(np.int64)).to(device)   # shape (k, n)
Hm = torch.from_numpy(Hm_np.astype(np.int64)).to(device)   # shape (4M, 8M)

# choose desired_k (must be <= Gm.shape[0])
desired_k = 14
# Select random full-rank rows from Gm (do selection on CPU's numpy for ease)
def select_full_rank_rows_np(Gm_np, desired_k):
    nrows = Gm_np.shape[0]
    for _ in range(2000):
        idx = np.random.choice(nrows, desired_k, replace=False)
        Gr = Gm_np[idx, :].copy().astype(np.uint8)
        if gf2_rank(Gr) == desired_k:
            return Gr, idx
    raise RuntimeError("Failed to select full-rank rows.")
Gr_np, selected_idx = select_full_rank_rows_np(Gm_np, desired_k)
Gr = torch.from_numpy(Gr_np.astype(np.int64)).to(device)   # shape (desired_k, n)
print("Gr Shape (torch) =", Gr.shape)
Hr = Hm  # here we keep Hm as is (converted to torch above)
print("Hr shape (torch) =", Hr.shape)

# ----------------------------
# Generate messages and codewords ON GPU (torch)
# ----------------------------
k = desired_k
n_messages = 2**k
# create messages matrix (n_messages x k) of bits, using torch on device
# Note: building full 2**k matrix on GPU is fine for k=14 (16384 x 14)
bits_idx = torch.arange(k-1, -1, -1, device=device).unsqueeze(0)  # [1,k]
msgs_int = torch.arange(n_messages, device=device).unsqueeze(1)    # [N,1]
messages = ((msgs_int >> bits_idx) & 1).to(torch.float32)         # [N, k], float32 of 0/1

# codewords: messages @ Gr (mod 2). Gr is (k, n) so do matmul then mod 2
# convert Gr to 0/1 integer type on GPU
Gr_bin = Gr.to(torch.int64)
messages_int = messages.to(torch.int64)
# codewords = torch.remainder(messages_int @ Gr_bin, 2).to(torch.int8)  # [N, n]
codewords = torch.remainder((messages.float() @ Gr_bin.float()), 2).to(torch.int8)

# convert to BPSK -1/+1 as float on device
codewords_bpsk = (2 * codewords.to(torch.float32) - 1.0)

n_samples = messages.shape[0]
# create train/test split on CPU (small)
idx = np.random.permutation(n_samples)
train_idx = idx[:int(0.5*n_samples)]
test_idx = idx[int(0.5*n_samples):]

# ----------------------------
# Dataset class (returns GPU tensors directly)
# ----------------------------
class MultiInputDataset(Dataset):
    def __init__(self, messages_t, codewords_bpsk_t, Hr_t, indices, variances, realizations_per_var=10, device=device):
        # messages_t and codewords_bpsk_t are torch tensors already on device
        self.messages = messages_t[indices].to(device)       # float32 [N_sel, k]
        self.codewords = codewords_bpsk_t[indices].to(device) # float32 [N_sel, n]
        # Hr matrices: we will prepare both binary and bipolar forms on device
        # Hr_t expected as torch int tensor on device (shape: m, n)
        Hr_bin = torch.remainder(Hr_t.to(torch.int64), 2).to(torch.int64)   # [m, n]
        # Hr_bip: (2*Hr-1) mapped to float for inner products
        Hr_bip = (2 * (Hr_bin.to(torch.float32)) - 1.0).to(torch.float32)  # [m, n]
        # we want Hr_bin and Hr_bip in shapes used below. We'll store their transposes for matmul
        self.Hr_bin_T = Hr_bin.t().to(device)   # shape [n, m]
        self.Hr_bip = Hr_bip.to(device)         # shape [m, n]
        self.variances = list(variances)
        self.realizations_per_var = realizations_per_var
        # precompute sample-index list (small)
        self.samples = [(i, var) for i in range(len(indices)) for var in self.variances for _ in range(realizations_per_var)]
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, var = self.samples[idx]
        cw = self.codewords[i]   # already on device, float32
        # noise on GPU
        noise = torch.randn_like(cw, device=self.device) * (var**0.5)
        Y = cw + noise
        # per-sample normalization on GPU
        Y = (Y - Y.mean()) / (Y.std() + 1e-8)
        signY = torch.sign(Y)
        signY[signY == 0] = 1.0
        bits = ((signY + 1.0) / 2.0).float()  # 0/1 float
        # Hr_sign: bits @ Hr_bin_T  (bits [n], Hr_bin_T [n,m]) -> [m], mod 2
        Hr_sign = torch.remainder(bits @ self.Hr_bin_T.float(), 2.0)
        # HrY: Hr_bip @ Y  (m x n) @ (n) -> m
        HrY = self.Hr_bip @ Y
        HrY = (HrY - HrY.mean()) / (HrY.std() + 1e-8)
        target = self.messages[i]
        return Y, signY, Hr_sign, HrY, target

# ----------------------------
# Model (same architecture, on device)
# ----------------------------
class MultiBranchNet(nn.Module):
    def __init__(self, input_Y=128, input_HrY=64, input_signY=128, input_Hr_sign=64, out_bits=14):
        super().__init__()
        self.y_branch = nn.Sequential(nn.Linear(input_Y,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU())
        self.hry_branch = nn.Sequential(nn.Linear(input_HrY,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU())
        self.sign_branch = nn.Sequential(nn.Linear(input_signY,128), nn.Tanh(), nn.Linear(128,64), nn.Tanh())
        self.hr_sign_branch = nn.Sequential(nn.Linear(input_Hr_sign,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64+64+64+32,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,out_bits))

    def forward(self, Y, signY, Hr_sign, HrY):
        a = self.y_branch(Y)
        b = self.hry_branch(HrY)
        c = self.sign_branch(signY)
        d = self.hr_sign_branch(Hr_sign)
        x = torch.cat([a,b,c,d], dim=1)
        return self.fusion(x)

# ----------------------------
# Prepare datasets & loader (everything GPU)
# ----------------------------
# pick reasonable input sizes consistent with your Gr/Hr shapes:
n = codewords.shape[1]
m = Hr.shape[0]   # number of parity checks rows

# create dataset objects
train_dataset = MultiInputDataset(messages, codewords_bpsk, Hr, train_idx, variances=[0.1,0.2,0.3,0.4,0.5], realizations_per_var=50, device=device)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=False, num_workers=0)

# instantiate model, loss, optimizer on device
model = MultiBranchNet(input_Y=n, input_HrY=m, input_signY=n, input_Hr_sign=m, out_bits=k).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# Training
# ----------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
    for Y, signY, Hr_sign, HrY, target in loop:
        optimizer.zero_grad()
        logits = model(Y, signY, Hr_sign, HrY)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss/(loop.n + 1e-12))
    if (epoch % 5) == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

# ----------------------------
# BER computation (fully on GPU)
# ----------------------------
def compute_BER(model, messages_t, codewords_bpsk_t, Hr_t, indices, variances, batch_size=512, R=10, device=device):
    model.eval()
    # prepare Hr forms once (on device)
    Hr_bin = torch.remainder(Hr_t.to(torch.int64), 2).to(torch.int64)  # [m, n]
    Hr_bin_T = Hr_bin.t().to(device)   # [n, m]
    Hr_bip = (2 * Hr_bin.to(torch.float32) - 1.0).to(device)           # [m, n]
    ber_dict = {}
    with torch.no_grad():
        for var in variances:
            sigma = var**0.5
            total_bits = 0
            total_errors = 0
            for start in range(0, len(indices), batch_size):
                end = min(start+batch_size, len(indices))
                batch_ids = indices[start:end]
                msgs_batch = messages_t[batch_ids].to(device)           # [B, k], float
                cw_batch = codewords_bpsk_t[batch_ids].to(device)      # [B, n], float
                B = cw_batch.shape[0]
                for _ in range(R):
                    noise = torch.randn_like(cw_batch, device=device) * sigma
                    Y = cw_batch + noise
                    # normalize per sample (row-wise)
                    Y = (Y - Y.mean(dim=1, keepdim=True)) / (Y.std(dim=1, keepdim=True) + 1e-8)
                    signY = torch.sign(Y)
                    signY[signY == 0] = 1.0
                    bits = ((signY + 1.0) / 2.0).float()   # [B, n]
                    # Hr_sign: (bits @ Hr_bin_T.float()) % 2
                    Hr_sign = torch.remainder(bits @ Hr_bin_T.float(), 2.0)
                    # HrY: (Hr_bip @ Y.T).T  => we compute as (Y @ Hr_bip.T)
                    HrY = (Y @ Hr_bip.t())
                    HrY = (HrY - HrY.mean(dim=1, keepdim=True)) / (HrY.std(dim=1, keepdim=True) + 1e-8)
                    logits = model(Y, signY, Hr_sign, HrY)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    total_errors += (preds != msgs_batch).float().sum().item()
                    total_bits += msgs_batch.numel()
            ber = total_errors / total_bits
            ber_dict[var] = ber
            print(f"Variance {var}: BER={ber:.6e}")
    return ber_dict

# call compute_BER (uses GPU tensors)
variances_test = [0.1,0.2,0.3,0.4,0.5]
ber_test = compute_BER(model, messages, codewords_bpsk, Hr, test_idx, variances_test)

# move BER results to CPU for plotting
x = list(ber_test.keys())
y = [ber_test[v] for v in x]

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o')
plt.xlabel("Noise variance")
plt.ylabel("BER")
plt.title("BER vs Noise Variance (GPU)")
plt.grid(True)
plt.show()
