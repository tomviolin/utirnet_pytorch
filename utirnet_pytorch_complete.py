#!/usr/bin/env python3

"""
UTIRnet pipeline — Python / PyTorch
Complete translation of the MATLAB pipeline to Python+PyTorch.

Requirements:
  pip install torch torchvision numpy scipy pillow matplotlib tqdm

Usage:
  - Edit the PARAMETERS section below (paths, Z, lambda_um, dx, etc.)
  - Run: python utirnet_py.py
"""

import os, sys, shutil
import random
import math
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import savemat, loadmat
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.tensorboard import SummaryWriter


# -----------------------------
# ========= PARAMETERS ========
# -----------------------------
# optical params (same units as MATLAB snippet)
Z = 17000.0        # propagation distance in micrometers (um)
lambda_um = 0.405  # wavelength in um (405 nm)
dx = 2.4           # sampling/pixel size in object plane (um)

# dataset / image params
IMG_SIZE = 512    # target size for images (512x512)
TRAIN_N = 650      # number of train images
VAL_N = 60        # validation images
#TRAIN_N = 2      # number of train images
#VAL_N =  1        # validation images

# path to dataset (flowers dataset expected to have subfolders)
# change this to your local path
DATASET_PATH = r"../flowers"

# which class indices (1-based folder ordering) to use for train/val
TRAIN_CLASS_INDICES = [1, 4, 5]
VAL_CLASS_INDICES = [2, 3]

# training params
BATCH_SIZE = 1
#EPOCHS = 30
EPOCHS = 55
LR = 1e-4
if "--cpu" in sys.argv:
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("***Using device:", DEVICE)
# save paths
# YYMMDD_HHMMSS formatted now string
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
OUT_DIR_DATED = "UTIRnet_output_"+timestamp
os.makedirs(OUT_DIR_DATED, exist_ok=True)
OUT_DIR_DUMP = "UTIRnet_output"
os.makedirs(OUT_DIR_DUMP, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir=".")
#os.system("tensorboard --logdir . &")

# -----------------------------
# ==== Utility / Optics code ==
# -----------------------------
def _to_float_img(im, size=IMG_SIZE):
    """Load PIL image or numpy array to normalized float32 0..1 size (H,W)."""
    if isinstance(im, Image.Image):
        im = im.convert("L")
        im = im.resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32)  / 255.0
        #arr -= arr.min()
        #arr = arr / arr.max()
    else:
        # assume numpy
        arr = np.asarray(im, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        arr = np.asarray(arr, dtype=np.float32) / 255.0
        #arr -= arr.min()
        #arr = arr / arr.max()
    return arr


def angular_spectrum_prop(u0, dx, wavelength, z):
    """
    Angular spectrum propagation of field u0 (2D complex numpy array)
    dx: sampling interval (same units as wavelength and z)
    wavelength: same units (um)
    z: propagation distance (um) — can be negative for backpropagation
    Returns propagated field u1 (2D complex numpy array)
    """
    # ensure complex
    u0 = np.asarray(u0, dtype=np.complex64)
    ny, nx = u0.shape
    k = 2 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * z * 2 * np.pi * np.sqrt(np.maximum(0.0, (1.0 / wavelength**2) - (FX**2 + FY**2))))
    # evanescent components handled by sqrt(max(0,...))
    U0 = np.fft.fft2(u0)
    U1 = U0 * H
    u1 = np.fft.ifft2(U1)
    return u1


# -----------------------------
# ==== Hologram generation ====
# -----------------------------
def GenerateHologram(img, Z_um, wavelength_um, dx_um, mode="amp"):
    """
    Given an input image (2D float 0..1), generate (input, GT, holo) triplet:
      - input: complex field backpropagated from approximated sensor amplitude (this includes twin image)
               returned as complex numpy array Uin (object plane)
      - GT: ground truth (float), either amplitude image (mode='amp') or phase image mapped to [-pi,pi] (mode='phs')
      - holo: intensity hologram measured at sensor plane (2D float)
    Steps:
      1) build object complex field O(x,y) = A(x,y) * exp(1j * phi(x,y))
      2) propagate forward by +Z to sensor (U_sensor)
      3) holo = |U_sensor|^2
      4) approximate sensor field as sqrt(holo) * exp(1j*0)
      5) backpropagate approx sensor field by -Z to object plane -> Uin (contains twin-image)
    """
    # Normalize input image
    im = _to_float_img(img, size=IMG_SIZE)

    # define target amplitude or phase
    if mode == "amp":
        A = im.copy()  # amplitude target in 0..1
        phi = np.zeros_like(A)
        GT = A.copy()
    elif mode == "phs":
        # map input image intensities to phase in [0, 2pi), then shift to center (similar to MATLAB's GT-pi)
        phi = (im * 2.0 * np.pi) - np.pi   # in [-pi, pi]
        A = np.ones_like(phi) * 0.5        # some constant amplitude (or use image brightness)
        GT = phi.copy()
    else:
        raise ValueError("mode must be 'amp' or 'phs'")

    # object field
    O = A * np.exp(1j * phi)

    # forward propagate to sensor
    U_sensor = angular_spectrum_prop(O, dx_um, wavelength_um, Z_um)

    # hologram intensity at sensor
    holo = np.abs(U_sensor) ** 2

    # approximate sensor complex field (amplitude = sqrt(holo), phase = 0)
    approx_sensor_field = np.sqrt(holo) * np.exp(1j * 0.0)

    # backpropagate to object plane -> this contains twin-image
    Uin = angular_spectrum_prop(approx_sensor_field, dx_um, wavelength_um, -Z_um)

    # return input (complex), GT (float), holo (float)
    return Uin, GT, holo


# -----------------------------
# === Dataset generator (folder) ===
# -----------------------------
def _list_class_images(root, class_indices):
    """
    Return list of file paths for images in specified class indices (1-based).
    Classes are determined by subdirectories under root (alphabetical order).
    """
    subdirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    picked = []
    for ci in class_indices:
        idx = ci - 1
        if idx < 0 or idx >= len(subdirs):
            continue
        dpath = os.path.join(root, subdirs[idx])
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            files.extend(glob(os.path.join(dpath, ext)))
        picked.extend(files)
    return picked


def GenerateDataset(path, class_indices, n_images, Z_um, wavelength_um, dx_um, mode="amp",validate=False):
    """
    Create dataset of n_images from images inside class directories in path.
    Returns numpy arrays:
      inputs: (n, 1, H, W) float32 -> amplitude or phase of Uin (depends on mode)
      targets: (n, 1, H, W) float32 -> GT amplitude or GT phase
      holos: (n, H, W) float32 -> hologram intensity
    NOTE: generating dataset is time-consuming; consider saving to disk after generation.
    """
    np.random.seed(0)
    file_list = _list_class_images(path, class_indices)
    if len(file_list) == 0:
        raise RuntimeError(f"No images found in {path} for classes {class_indices}")

    inputs = []
    targets = []
    holos = []

    for i in tqdm(range(n_images), desc=f"GenerateDataset({mode})"):
        f = random.choice(file_list)
        if (np.random.rand() < .0 and not validate):
            # random dots pattern
            img = Image.fromarray((np.random.rand(IMG_SIZE, IMG_SIZE) > 0.7).astype(np.float32))
        else:
            img = Image.open(f)

        Uin, GT, holo = GenerateHologram(img, Z_um, wavelength_um, dx_um, mode=mode)

        # prepare network inputs (amplitude or phase derived from Uin)
        if mode == "amp":
            inp = np.abs(Uin)
            tar = GT  # amplitude target
        else:
            inp = np.angle(Uin)
            tar = GT  # phase target

        # normalize inputs & targets properly to stable numeric range
        # For amplitude: range 0..max -> scale to [0,1] by dividing by max of inp
        if mode == "amp":
            # avoid divide by zero
            m = inp.max() if inp.max() > 0 else 1.0
            inp = inp / m
            # target A is already 0..1
            tar = tar.astype(np.float32)
        else:
            # for phase, keep range [-pi, pi]
            inp = inp.astype(np.float32)
            tar = tar.astype(np.float32)

        inputs.append(inp.astype(np.float32)[None, ...])   # (1,H,W)
        targets.append(tar.astype(np.float32)[None, ...])
        holos.append(holo.astype(np.float32))

        # early stop if file count small and requested many samples
        if len(file_list) == 1 and i > 10000:
            break

    inputs = np.stack(inputs, axis=0)   # (n,1,H,W)
    targets = np.stack(targets, axis=0) # (n,1,H,W)
    holos = np.stack(holos, axis=0)     # (n,H,W)

    return inputs, targets, holos


# -----------------------------
# ===== Network Architecture ==
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.final(d1)
        return out


# -----------------------------
# ===== Training / Helpers ====
# -----------------------------
def make_loader(np_inputs, np_targets, batch_size=BATCH_SIZE):
    t_in = torch.from_numpy(np_inputs)     # shape (n,1,H,W)
    t_ta = torch.from_numpy(np_targets)
    ds = TensorDataset(t_in, t_ta)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader


def train_once(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {'train_loss': [], 'val_loss': []}
    for ep in range(epochs):
        model.train()
        tloss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            train_count += 1
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            print(f"[Ep {ep+1}/{epochs}] [{train_count}/{len(train_loader)}] [Batch] batch_loss={loss.item():.6f} tloss={tloss:.6f}")

        tloss /= len(train_loader)

        # validation
        model.eval()
        vloss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                vloss += criterion(out, yb).item()
                print(f"[Ep {ep+1}/{epochs}] [{val_count+1}/{len(val_loader)}] [Val Batch] batch_vloss={criterion(out, yb).item():.6f}")
                val_count += 1
        vloss /= len(val_loader)

        scheduler.step()
        history['train_loss'].append(tloss)
        history['val_loss'].append(vloss)
        writer.add_scalar('Loss/Train', tloss, ep+1)
        writer.add_scalar('Loss/Val', vloss, ep+1)
        writer.flush()
        print(f"[Epoch {ep+1}/{epochs}] train={tloss:.6f} val={vloss:.6f}")

    return history


# -----------------------------
# ===== UTIRnet Reconstruction =
# -----------------------------
def UTIRnetReconstruction(holo, CNN_A, CNN_P, Z_um, wavelength_um, dx_um, m1=None, show_debug=0):
    """
    holo: 2D numpy array (hologram intensity), optionally padded (float)
    CNN_A, CNN_P: trained PyTorch models (in eval mode or will be set to eval)
    Returns:
      Yout: complex reconstructed object (numpy complex) formed from Yamp*exp(1j*Yphs)
      Yamp: amplitude reconstruction (numpy float)
      Yphs: phase reconstruction (numpy float - radians)
      Uout: input field backpropagated from approximate sensor (numpy complex)
    Steps:
      - approximate sensor field = sqrt(holo) * exp(1j*0)
      - backpropagate to object plane => Uout (contains twin-image)  -- this mirrors MATLAB code's input AS
      - prepare CNN inputs: amplitude = abs(Uout) normalized; phase = angle(Uout)
      - run CNN_A on amplitude input, CNN_P on phase input -> Yamp_pred, Yphs_pred
      - combine: Yout = Yamp_pred * exp(1j * Yphs_pred)
    """
    # ensure numpy arrays
    holo = np.asarray(holo, dtype=np.float32)
    approx_sensor = np.sqrt(holo) * np.exp(1j * 0.0)

    # backpropagate to object plane (Uout)
    Uout = angular_spectrum_prop(approx_sensor, dx_um, wavelength_um, -Z_um)

    # prepare inputs for CNNs
    amp_in = np.abs(Uout).astype(np.float32)
    phs_in = np.angle(Uout).astype(np.float32)

    # normalization — for amplitude we scale to [0,1]
    m = amp_in.max() if amp_in.max() > 0 else 1.0
    amp_in_n = amp_in / m

    # convert to tensors (N=1, C=1, H, W)
    A_tensor = torch.from_numpy(amp_in_n[None, None, ...]).to(DEVICE)
    P_tensor = torch.from_numpy(phs_in[None, None, ...]).to(DEVICE)

    CNN_A = CNN_A.to(DEVICE)
    CNN_P = CNN_P.to(DEVICE)
    CNN_A.eval(); CNN_P.eval()
    with torch.no_grad():
        Yamp_pred = CNN_A(A_tensor).cpu().numpy().squeeze(0).squeeze(0)
        Yphs_pred = CNN_P(P_tensor).cpu().numpy().squeeze(0).squeeze(0)

    # if amplitude network was trained with normalized amplitude, rescale
    # Here we assume target amplitudes were 0..1 (so direct)
    Yamp = Yamp_pred.astype(np.float32)
    Yamp -= Yamp.min()
    #Yamp = Yamp / Yamp.max() # * m  # rescale to original max amplitude
    Yphs = Yphs_pred.astype(np.float32) / 255.0

    Yout = Yamp * np.exp(1j * Yphs)

    if show_debug:
        print("Uout amplitude range:", amp_in.min(), amp_in.max())
        print("Yamp range:", Yamp.min(), Yamp.max())

    return Yout, Yamp, Yphs, Uout


# -----------------------------
# ======= Main pipeline =======
# -----------------------------
def main():
    # ---------------------
    # Generate / load dataset
    # ---------------------
    if not os.path.exists(os.path.join(OUT_DIR_DUMP, "train_dataset.npz")) or \
            "--regenerate-dataset" in sys.argv or \
            '--clean' in sys.argv:
        print("Generating datasets (this may take a while)...")
        # You can also create and save dataset to disk once and load later to avoid regenerating each run.
        inTrAmp, tarTrAmp, holosTrAmp = GenerateDataset(DATASET_PATH, TRAIN_CLASS_INDICES, TRAIN_N, Z, lambda_um, dx, mode='amp')
        inTrPhs, tarTrPhs, holosTrPhs = GenerateDataset(DATASET_PATH, TRAIN_CLASS_INDICES, TRAIN_N, Z, lambda_um, dx, mode='phs')

        np.savez_compressed(os.path.join(OUT_DIR_DUMP, "train_dataset.npz"),
            inTrAmp=inTrAmp, tarTrAmp=tarTrAmp, holosTrAmp=holosTrAmp,
            inTrPhs=inTrPhs, tarTrPhs=tarTrPhs, holosTrPhs=holosTrPhs
        )
        shutil.copyfile(os.path.join(OUT_DIR_DUMP, "train_dataset.npz"), os.path.join(OUT_DIR_DATED, "train_dataset.npz"))
    else:
        print("Loading existing datasets from disk...")
        srcdir = open(os.path.join(OUT_DIR_DUMP, "last_output_dir.txt"), "r").read().strip()
        os.symlink(os.path.join("../"+srcdir, "train_dataset.npz"), os.path.join(OUT_DIR_DATED, "train_dataset.npz"))
        data = np.load(os.path.join(OUT_DIR_DUMP, "train_dataset.npz"))

        inTrAmp = data['inTrAmp']
        tarTrAmp = data['tarTrAmp']
        holosTrAmp = data['holosTrAmp']
        inTrPhs = data['inTrPhs']
        tarTrPhs = data['tarTrPhs']
        holosTrPhs = data['holosTrPhs']

    if not os.path.exists(os.path.join(OUT_DIR_DUMP, "val_dataset.npz")) or \
            "--regenerate-dataset" in sys.argv or \
            '--clean' in sys.argv:

        inValAmp, tarValAmp, holosValAmp = GenerateDataset(DATASET_PATH, VAL_CLASS_INDICES, VAL_N, Z, lambda_um, dx, mode='amp', validate=True)
        inValPhs, tarValPhs, holosValPhs = GenerateDataset(DATASET_PATH, VAL_CLASS_INDICES, VAL_N, Z, lambda_um, dx, mode='phs', validate=True)

        np.savez_compressed(os.path.join(OUT_DIR_DUMP, "val_dataset.npz"),
            inValAmp=inValAmp, tarValAmp=tarValAmp, holosValAmp=holosValAmp,
            inValPhs=inValPhs, tarValPhs=tarValPhs, holosValPhs=holosValPhs
        )
        shutil.copyfile(os.path.join(OUT_DIR_DUMP, "val_dataset.npz"), os.path.join(OUT_DIR_DATED, "val_dataset.npz"))

    else:
        print("Loading existing datasets from disk...")
        srcdir = open(os.path.join(OUT_DIR_DUMP, "last_output_dir.txt"), "r").read().strip()
        os.symlink(os.path.join("../"+srcdir, "val_dataset.npz"), os.path.join(OUT_DIR_DATED, "val_dataset.npz"))
        data = np.load(os.path.join(OUT_DIR_DUMP, "val_dataset.npz"))

        inValAmp = data['inValAmp']
        tarValAmp = data['tarValAmp']
        holosValAmp = data['holosValAmp']
        inValPhs = data['inValPhs']
        tarValPhs = data['tarValPhs']
        holosValPhs = data['holosValPhs']


    """
    # optionally save generated datasets (uncomment)
    savemat(os.path.join(OUT_DIR, "datasets.mat"),
            {"inTrAmp": inTrAmp, "tarTrAmp": tarTrAmp, "holosTrAmp": holosTrAmp,
             "inTrPhs": inTrPhs, "tarTrPhs": tarTrPhs, "holosTrPhs": holosTrPhs,
             "inValAmp": inValAmp, "tarValAmp": tarValAmp, "holosValAmp": holosValAmp,
             "inValPhs": inValPhs, "tarValPhs": tarValPhs, "holosValPhs": holosValPhs})
    """

    # ---------------------
    # Prepare loaders
    # ---------------------
    train_loader_A = make_loader(inTrAmp, tarTrAmp, BATCH_SIZE)
    val_loader_A = make_loader(inValAmp, tarValAmp, BATCH_SIZE)

    train_loader_P = make_loader(inTrPhs, tarTrPhs, BATCH_SIZE)
    val_loader_P = make_loader(inValPhs, tarValPhs, BATCH_SIZE)

    # ---------------------
    # Build networks
    # ---------------------
    CNN_A = UNet(in_ch=1, out_ch=1)
    CNN_P = UNet(in_ch=1, out_ch=1)

    # -----------------------------
    # load CNN_A pretrained weights if available
    # -----------------------------

    pretrained_A_path = os.path.join(OUT_DIR_DUMP, "CNN_A_state.pth")
    if os.path.exists(pretrained_A_path) and not "--retrain" in sys.argv and \
            not '--clean' in sys.argv:
            
        print("Loading pretrained CNN_A weights...")
        srcdir = open(os.path.join(OUT_DIR_DUMP, "last_output_dir.txt"), "r").read().strip()
        os.symlink(os.path.join('../'+srcdir, "CNN_A_state.pth"), os.path.join(OUT_DIR_DATED, "CNN_A_state.pth"))
        info_A = CNN_A.load_state_dict(torch.load(pretrained_A_path, map_location=DEVICE))

    if "--continue-train" in sys.argv:
        # ---------------------
        # Train CNN_A
        # ---------------------
        print("Training CNN_A (amplitude net)...")
        info_A = train_once(CNN_A, train_loader_A, val_loader_A, epochs=EPOCHS, lr=LR, device=DEVICE)
        torch.save(CNN_A.state_dict(), os.path.join(OUT_DIR_DATED, "CNN_A_state.pth"))
        shutil.copyfile(os.path.join(OUT_DIR_DATED, "CNN_A_state.pth"),
                os.path.join(OUT_DIR_DUMP, "CNN_A_state.pth"))

    # -----------------------------
    # load CNN_P pretrained weights if available
    # -----------------------------
    pretrained_P_path = os.path.join(OUT_DIR_DUMP, "CNN_P_state.pth")
    if os.path.exists(pretrained_P_path) and not "--retrain" in sys.argv and \
            not '--clean' in sys.argv:
        print("Loading pretrained CNN_P weights...")
        srcdir = open(os.path.join(OUT_DIR_DUMP, "last_output_dir.txt"), "r").read().strip()
        os.symlink(os.path.join('../'+srcdir, "CNN_P_state.pth"), os.path.join(OUT_DIR_DATED, "CNN_P_state.pth"))
        info_P = CNN_P.load_state_dict(torch.load(pretrained_P_path, map_location=DEVICE))
    
    if "--continue-train" in sys.argv:

        # ---------------------
        # Train CNN_P
        # ---------------------
        print("Training CNN_P (phase net)...")
        info_P = train_once(CNN_P, train_loader_P, val_loader_P, epochs=EPOCHS, lr=LR, device=DEVICE)
        torch.save(CNN_P.state_dict(), os.path.join(OUT_DIR_DATED, "CNN_P_state.pth"))
        shutil.copyfile(os.path.join(OUT_DIR_DATED, "CNN_P_state.pth"),
                os.path.join(OUT_DIR_DUMP, "CNN_P_state.pth"))

    # ---------------------
    # Save metadata
    # ---------------------
    metadata = {
        "Z_mm": Z / 1000.0,
        "lambda_um": lambda_um,
        "dx_um": dx,
        "training_info_A": info_A,
        "training_info_P": info_P,
    }
    savemat(os.path.join(OUT_DIR_DATED, f"UTIRnet_meta_Z-{Z/1000:.2f}mm_dx-{dx}_lambda-{int(lambda_um*1000)}nm.mat"),
            metadata)
    shutil.copyfile(os.path.join(OUT_DIR_DATED, f"UTIRnet_meta_Z-{Z/1000:.2f}mm_dx-{dx}_lambda-{int(lambda_um*1000)}nm.mat"),
            os.path.join(OUT_DIR_DUMP, f"UTIRnet_meta_Z-{Z/1000:.2f}mm_dx-{dx}_lambda-{int(lambda_um*1000)}nm.mat"))
    print(f"{OUT_DIR_DATED}",file=open(os.path.join(OUT_DIR_DUMP, "last_output_dir.txt"), "w")) 


    # reconstruct hologram file passed as argument
    #holo_file = "../flewers/type1/HOLO-gt.png"
    holo_file = "../flowers/rose/10090824183_d02c613f10_m.jpg"
    holo = _to_float_img(Image.open(holo_file), size=IMG_SIZE)
    plt.imshow(holo, cmap="gray")
    plt.savefig("junk.png", dpi=300)

    Yout, Yamp, Yphs, Uout = UTIRnetReconstruction(holo, CNN_A, CNN_P, Z, lambda_um, dx, m1=None, show_debug=1)
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.imshow(np.abs(Uout), cmap="gray", vmin=0, vmax=1.1)
    plt.title("Input AS amplitude (with twin-image)")
    plt.axis("off") 
    plt.subplot(1,3,2)
    plt.imshow(np.abs(Yout), cmap="gray", vmin=0, vmax=1.1)
    plt.title("UTIRnet amplitude reconstruction")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(np.angle(Yout), cmap="gray", vmin=-math.pi, vmax=math.pi)
    plt.title("UTIRnet phase reconstruction")
    plt.axis("off") 

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_DATED, f"UTIRnet_{timestamp}_holotest_reconstruction.png"), dpi=300)
    plt.show()


    exit(0)



    # ---------------------
    # Example: reconstruct one validation hologram (index 1)
    # ---------------------
    imNo = np.random.randint(0, VAL_N)
    for AmpPhs in [1, 2]:
        # AmpPhs = 0  # 1 amplitude, 2 phase
        if AmpPhs == 1:
            GT = tarValAmp[imNo, 0, :, :]
            holo = holosValAmp[imNo]
            ap_name = "amplitude"
        else:
            GT = tarValPhs[imNo, 0, :, :]
            holo = holosValPhs[imNo]
            ap_name = "phase"

        # pad hologram like MATLAB does:
        pad = 4 #IMG_SIZE // 4 
        holoP = np.pad(holo, ((pad, pad), (pad, pad)), mode="edge")

        # Reconstruction
        Yout, Yamp, Yphs, Uout = UTIRnetReconstruction(holoP, CNN_A, CNN_P, Z, lambda_um, dx, m1=None, show_debug=1)

        # remove padding
        Yout = Yout[pad:-pad, pad:-pad]
        Uout = Uout[pad:-pad, pad:-pad]
        Yamp = Yamp[pad:-pad, pad:-pad]
        Yphs = Yphs[pad:-pad, pad:-pad]

        # visualize like in MATLAB
        #plt.figure(figsize=(10, 8))

        fig,axs = plt.subplots(1, 3, figsize=(15,5))
        if AmpPhs == 1:
            rng = (0.0, 1.1)
            #plt.subplot(1, 3, 1)

            im = axs[0].imshow(np.abs(Uout), cmap="gray", vmin=rng[0], vmax=rng[1]); plt.title("input AS amplitude (with twin-image)")
            fig.colorbar(im,ax=axs[0])
            #writer.add_figure(f'Reconstruction/Input_{ap_name}', plt.gcf(), global_step=0)
            #plt.axis("off")
            #plt.subplot(1, 3, 2)
            im = axs[1].imshow(np.abs(Yout), cmap="gray", vmin=rng[0], vmax=rng[1]); plt.title("UTIRnet amplitude reconstruction")
            fig.colorbar(im,ax=axs[1])
            #writer.add_figure(f'Reconstruction/UTIRnet_{ap_name}', plt.gcf(), global_step=0)
            #plt.axis("off")
            #plt.subplot(1, 3, 3)
            im = axs[2].imshow(GT, cmap="gray", vmin=rng[0], vmax=rng[1]); plt.title("Ground truth amplitude")
            fig.colorbar(im,ax=axs[2])
            #writer.add_figure(f'Reconstruction/GroundTruth_{ap_name}', plt.gcf(), global_step=0)
            #plt.axis("off")
        else:
            rng = (-math.pi, math.pi)
            #plt.subplot(1, 3, 1)
            im = axs[0].imshow(np.angle(Uout), cmap="gray", vmin=rng[0], vmax=rng[1]); plt.title("input AS phase (with twin-image)")
            fig.colorbar(im,ax=axs[0])
            #writer.add_figure(f'Reconstruction/Input_{ap_name}', plt.gcf(), global_step=0)
            #plt.axis("off")
            #plt.subplot(1, 3, 2)
            im = axs[1].imshow(np.angle(Yout), cmap="gray", vmin=rng[0], vmax=rng[1]); plt.title("UTIRnet reconstruction")
            fig.colorbar(im,ax=axs[1])
            #writer.add_figure(f'Reconstruction/UTIRnet_{ap_name}', plt.gcf(), global_step=0)
            #plt.axis("off")
            #plt.subplot(1, 3, 3)
            im = axs[2].imshow(GT - math.pi, cmap="gray", vmin=rng[0], vmax=rng[1]); plt.title("Ground truth (shifted)")
            fig.colorbar(im,ax=axs[2])
            #writer.add_figure(f'Reconstruction/GroundTruth_{ap_name}', plt.gcf(), global_step=0)
            #plt.axis("off")
        #plt.tight_layout()
        #writer.add_figure(f'Reconstruction/{ap_name}', plt.gcf(), global_step=0)
        #writer.flush()
        plt.savefig(os.path.join(OUT_DIR_DATED, f"UTIRnet_{timestamp}_{ap_name}_reconstruction_example.png"), dpi=300)
        plt.close('all')
        plt.show()


if __name__ == "__main__":
    main()
