"""Honeypot attack GAN — learns attack-session distribution, generates synthetic attacks."""
import torch, torch.nn as nn, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["login_attempts","login_success","login_failed",
            "cmd_count","cmd_failed","downloads","duration_ms"]

def load_attacks(path="honeypot_features.csv"):
    df = pd.read_csv(path)
    attacks = df[df["label"] == 1][FEATURES].values.astype(np.float32)
    scaler = StandardScaler().fit(attacks)
    return scaler.transform(attacks), scaler

class Generator(nn.Module):
    def __init__(self, z=16, out=len(FEATURES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z,32), nn.ReLU(),
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,out))
    def forward(self,x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, inp=len(FEATURES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp,64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64,32), nn.LeakyReLU(0.2),
            nn.Linear(32,1), nn.Sigmoid())
    def forward(self,x): return self.net(x)

def train_gan(data, epochs=300, z=16, batch=16):
    G, D = Generator(z), Discriminator()
    optG = torch.optim.Adam(G.parameters(), lr=2e-4)
    optD = torch.optim.Adam(D.parameters(), lr=2e-4)
    bce = nn.BCELoss()
    data_t = torch.tensor(data)
    for ep in range(epochs):
        idx = torch.randint(0, len(data_t), (batch,))
        real = data_t[idx]
        # train D
        z_noise = torch.randn(batch, z)
        fake = G(z_noise).detach()
        lossD = bce(D(real), torch.ones(batch,1)) + bce(D(fake), torch.zeros(batch,1))
        optD.zero_grad(); lossD.backward(); optD.step()
        # train G
        z_noise = torch.randn(batch, z)
        gen = G(z_noise)
        lossG = bce(D(gen), torch.ones(batch,1))
        optG.zero_grad(); lossG.backward(); optG.step()
        if (ep+1) % 50 == 0:
            print(f"epoch {ep+1:4d} | D {lossD.item():.3f} | G {lossG.item():.3f}")
    return G

def generate(G, scaler, n=100, z=16):
    with torch.no_grad():
        synth = G(torch.randn(n, z)).numpy()
    return scaler.inverse_transform(synth)

if __name__ == "__main__":
    data, scaler = load_attacks()
    print(f"Training GAN on {len(data)} real attack sessions...")
    G = train_gan(data)
    synth = generate(G, scaler, n=100)
    out = pd.DataFrame(synth, columns=FEATURES)
    out["label"] = 1
    out.to_csv("synthetic_attacks.csv", index=False)
    print(f"\nGenerated 100 synthetic attacks -> synthetic_attacks.csv")
    print(out.head())
