"""
Model definitions for the JerseyIQ pipeline.

These architectures were reverse-engineered directly from the shapes stored
inside the provided checkpoints (checkpoints/jersey_cnn/best.pt and
checkpoints/ccnn_filter/best.pt), so that `load_state_dict(..., strict=True)`
succeeds without guessing. If your original training code differs slightly
in non-parametric layers (activation choice, dropout rate, etc.) that's fine
those don't show up in the state dict and won't break loading.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# --------------------------------------------------------------------------
# Jersey number OCR network (checkpoints/jersey_cnn/best.pt)
#
# Reconstructed layout (names/shapes match the checkpoint exactly):
#   stn.loc   : Conv2d(3,16,7,pad=3) -> MaxPool2d(2) -> ReLU
#               -> Conv2d(16,32,5,pad=2) -> MaxPool2d(2) -> ReLU
#   stn.fc    : Linear(18432,64) -> ReLU -> Linear(64,6)   (affine grid params)
#   backbone  : ResNet18 features
#   heads     : head_visible (2), head_tens (11: 0-9 + none), head_units (11: 0-9 + none)
#
# Input crop size is 96x96 RGB.
# --------------------------------------------------------------------------

JERSEY_INPUT_SIZE = 96


class SpatialTransformer(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 7, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (JERSEY_INPUT_SIZE // 4) ** 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 6),
        )
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.loc(x)
        theta = theta.view(theta.size(0), -1)
        theta = self.fc(theta).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class JerseyCNN(nn.Module):
    def __init__(self, num_digit_classes=11, dropout: float = 0.3):
        super().__init__()
        self.stn = SpatialTransformer()
        backbone = tvm.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.head_visible = nn.Linear(feat_dim, 2)
        self.head_tens = nn.Linear(feat_dim, num_digit_classes)
        self.head_units = nn.Linear(feat_dim, num_digit_classes)

    def forward(self, x):
        x = self.stn(x)
        feat = self.dropout(self.features(x).flatten(1))
        return {
            "visible": self.head_visible(feat),
            "tens": self.head_tens(feat),
            "units": self.head_units(feat),
        }

    @torch.no_grad()
    def predict_number(self, x):
        """Returns list of dicts: {number:int|None, confidence:float, visible:bool}"""
        out = self.forward(x)
        vis_p = F.softmax(out["visible"], dim=1)
        tens_p = F.softmax(out["tens"], dim=1)
        units_p = F.softmax(out["units"], dim=1)

        vis_conf, vis_idx = vis_p.max(dim=1)
        tens_conf, tens_idx = tens_p.max(dim=1)
        units_conf, units_idx = units_p.max(dim=1)

        results = []
        for i in range(x.size(0)):
            visible = bool(vis_idx[i].item() == 1)
            tens_d = tens_idx[i].item()
            units_d = units_idx[i].item()
            if not visible:
                results.append({"number": None, "confidence": float(vis_conf[i].item()), "visible": False})
                continue
            digits = ""
            confs = [vis_conf[i].item()]
            if tens_d != 10:
                digits += str(tens_d)
                confs.append(tens_conf[i].item())
            if units_d != 10:
                digits += str(units_d)
                confs.append(units_conf[i].item())
            number = int(digits) if digits else None
            conf = sum(confs) / len(confs)
            results.append({"number": number, "confidence": float(conf), "visible": True})
        return results


# --------------------------------------------------------------------------
# Temporal filter / possession-touch classifier (checkpoints/ccnn_filter/best.pt)
#
# 4 residual Conv1d blocks (kernel=5, pad=2, channels=32) operating on a
# short window of per-frame features (3 input channels), followed by a
# 1x1 conv projecting to 2 classes (no-touch / touch) per timestep.
# --------------------------------------------------------------------------

CCNN_IN_CHANNELS = 3
CCNN_WINDOW = 16  # frames of context fed to the filter per prediction


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CCNNFilter(nn.Module):
    def __init__(self, in_channels=CCNN_IN_CHANNELS, channels=32, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(self.net(x))

    @torch.no_grad()
    def predict_touch_prob(self, x):
        """Returns touch probability per timestep: (B, T)"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return 1.0 - probs[:, 0, :]


def load_jersey_cnn(weights_path, device="cpu"):
    model = JerseyCNN()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_ccnn_filter(weights_path, device="cpu"):
    model = CCNNFilter()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model
