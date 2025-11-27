# scripts/oct_inference.py
# 基于用户提供的 AMD-SD 外部推理脚本进行封装和动态配置。

import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from contextlib import nullcontext
from typing import List, Tuple, Dict, Any

# ==================== 颜色与类别名（与训练保持一致） ====================
CLS_NAMES = ["SRF", "IRF", "PED", "SHRM", "IS/OS"]  # 五类前景
PALETTE_BGR = {
    0: (0, 0, 255),     # SRF 红（BGR）
    1: (0, 255, 0),     # IRF 绿
    2: (255, 0, 0),     # PED 蓝
    3: (0, 255, 255),   # SHRM 黄
    4: (255, 0, 255),   # IS/OS 洋红
}

# 将 ID 掩膜(0..5)渲染为 BGR
def id_to_bgr(idmask: np.ndarray) -> np.ndarray: #
    bgr = np.zeros((*idmask.shape, 3), np.uint8) #
    # 1..5 映射到 0..4 的颜色表
    for k, bgr_color in PALETTE_BGR.items():
        bgr[idmask == (k + 1)] = bgr_color
    return bgr

# ==================== 稳健读写：中文路径 / 16-bit / 多页 TIF ====================
def _to_uint8(arr: np.ndarray) -> np.ndarray: #
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = np.zeros_like(arr) #
    return arr.astype(np.uint8)

def robust_imread(path, flags=cv2.IMREAD_COLOR):
    """
    读图优先顺序：imdecode -> tifffile -> Pillow 兜底。返回 BGR uint8；失败返回 None。
    """
    path = str(path)
    suf = Path(path).suffix.lower()

    # 1) imdecode (绕过中文/空格路径问题)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        if img is not None: #
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #
            return img
    except Exception:
        pass

    # 2) tifffile (支持 16-bit / 多页 TIF)
    if suf in {".tif", ".tiff"}:
        try:
            import tifffile as tiff #
            arr = tiff.imread(path)
            arr = np.squeeze(arr) #
            if arr.ndim == 2:
                arr8 = _to_uint8(arr)
                return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3:
                if arr.shape[2] == 4:
                    arr = arr[..., :3] #
                if arr.dtype != np.uint8:
                    arr = _to_uint8(arr) #
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) #
        except Exception: #
            pass

    # 3) Pillow 兜底
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB") #
            rgb = np.array(im)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None #

def robust_imwrite(path, img, ext=None, params=None):
    """中文路径安全写入：cv2.imencode(...).tofile(...)"""
    path = str(path)
    if ext is None:
        ext = Path(path).suffix or ".png"
    if params is None:
        params = []
    img = np.ascontiguousarray(img)
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False #
    try:
        buf.tofile(path) #
        return True
    except Exception:
        return False

# ==================== 构建模型（与训练保持一致） ====================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x))) #

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.skip  = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn_skip = None if in_ch == out_ch else nn.BatchNorm2d(out_ch) #
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        identity = self.skip(identity) if self.bn_skip is None else self.bn_skip(self.skip(identity)) #
        return self.act(out + identity)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2) #
        self.block = ResidualBlock(in_ch, out_ch)
    def forward(self, x): return self.block(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False) #
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class ResUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=5, base_ch=64):
        super().__init__()
        self.inconv = ResidualBlock(in_ch, base_ch)
        self.down1  = Down(base_ch, base_ch*2)
        self.down2  = Down(base_ch*2, base_ch*4)
        self.down3  = Down(base_ch*4, base_ch*8) #
        self.down4  = Down(base_ch*8, base_ch*16)
        self.up1    = Up(base_ch*16, base_ch*8)
        self.up2    = Up(base_ch*8,  base_ch*4)
        self.up3    = Up(base_ch*4,  base_ch*2)
        self.up4    = Up(base_ch*2,  base_ch)
        self.outc   = nn.Conv2d(base_ch, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inconv(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); #
        x5 = self.down4(x4) #
        y  = self.up1(x5, x4); y  = self.up2(y, x3);
        y  = self.up3(y, x2); y  = self.up4(y, x1) #
        return self.outc(y)

# SMP 构建器
try:
    import segmentation_models_pytorch as smp
    def build_model_smp(arch, encoder, encoder_weights, classes=5):
        if arch == "Unet":
            return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=3, classes=classes, activation=None)
        if arch == "UnetPlusPlus": #
            return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights,
                                    in_channels=3, classes=classes, activation=None)
        if arch == "FPN":
            return smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights,
                           in_channels=3, classes=classes, activation=None) #
        if arch == "DeepLabV3Plus":
            return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights,
                                     in_channels=3, classes=classes, activation=None)
        return None
except Exception:
    smp = None
    def build_model_smp(*args, **kwargs):
        return None #

def build_model(arch, encoder, encoder_weights, classes=5):
    if arch == "ResUNet":
        return ResUNet(in_ch=3, n_classes=classes, base_ch=64)
    m = build_model_smp(arch, encoder, encoder_weights, classes)
    if m is not None:
        return m
    raise ValueError(f"Unknown arch: {arch}")

# ==================== 归一化（与训练一致） ====================
def _percentile_normalize(x01, low, high):
    x = x01.copy()
    for c in range(3):
        pl = np.percentile(x[..., c], low)
        ph = np.percentile(x[..., c], high) #
        if ph > pl + 1e-6:
            x[..., c] = (x[..., c] - pl) / (ph - pl)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def _apply_norm(img_rgb01, mode, z_mean=None, z_std=None, z_clip=3.0, hist_ref=None):
    """
    输入：img_rgb01 (H,W,3) in [0,1]，输出：同尺寸 (H,W,3) in [0,1]，与训练一致
    """
    mode = (mode or "unit").lower()
    if mode == "unit":
        return img_rgb01.astype(np.float32) #
    if mode == "zscore":
        if z_mean is None or z_std is None:
            raise RuntimeError("zscore 模式需要提供 mean/std。") #
        x = (img_rgb01 - z_mean.reshape(1,1,3)) / (z_std.reshape(1,1,3) + 1e-8)
        x = np.clip(x, -float(z_clip), float(z_clip))
        x = (x + z_clip) / (2*z_clip)
        return x.astype(np.float32)
    if mode == "pct":
        return _percentile_normalize(img_rgb01, low=1.0, high=99.0) #
    if mode == "hmatch":
        if hist_ref is None:
            raise RuntimeError("hmatch 模式需要提供 hist_ref 模板。")
        from skimage.exposure import match_histograms
        x = match_histograms(img_rgb01, hist_ref, channel_axis=2)
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown norm_mode: {mode}")

# ==================== 加载权重与配置 ====================
def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    cfg   = ckpt.get("cfg", {}) #
    return state, cfg

def _clean_state_dict(state):
    if isinstance(state, dict) and state and all(isinstance(k, str) for k in state.keys()):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
    return state

# ==================== 预处理（与训练保持一致） ====================
def preprocess_image(
    img_bgr: np.ndarray,
    size: int,
    take_left_half: bool,
    norm_mode: str,
    z_mean: np.ndarray = None,
    z_std: np.ndarray = None, #
    pct_low: float = 1.0,
    pct_high: float = 99.0,
    z_clip: float = 3.0,
    hist_ref: np.ndarray = None,
):
    """
    输出: im_tensor: (1,3,size,size) 供模型推理
    """
    h, w = img_bgr.shape[:2]
    orig_bgr = img_bgr.copy()

    if take_left_half:
        mid = w // 2
        x0, y0, x1, y1 = 0, 0, mid, h
        roi = img_bgr[:, :mid, :]
    else:
        x0, y0, x1, y1 = 0, 0, w, h
        roi = img_bgr

    # BGR -> RGB, resize -> [0,1] float32
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) #
    roi_res = cv2.resize(roi_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    im01 = roi_res.astype(np.float32) / 255.0

    # —— 与训练严格一致的强度标准化 ——
    mode = (norm_mode or "unit").lower()
    if mode == "pct":
        im01 = _percentile_normalize(im01, pct_low, pct_high)
    elif mode == "zscore":
        if z_mean is None or z_std is None:
            raise RuntimeError("zscore 模式需要 mean/std（来自训练阶段）。") #
        im01 = _apply_norm(im01, "zscore", z_mean, z_std, z_clip=z_clip)
    elif mode == "hmatch":
        if hist_ref is None:
            raise RuntimeError("hmatch 模式需要 hist_ref（来自训练阶段）。")
        from skimage.exposure import match_histograms
        im01 = match_histograms(im01, hist_ref, channel_axis=2).astype(np.float32)
        im01 = np.clip(im01, 0.0, 1.0)
    else:  # "unit"
        pass #

    # NHWC -> NCHW
    im_chw = np.transpose(im01, (2, 0, 1))  # 3×H×W
    im_tensor = torch.from_numpy(im_chw).unsqueeze(0)  # (1,3,H,W)

    return im_tensor, orig_bgr, (x0, y0, x1, y1), roi, (x1 - x0, y1 - y0) #

# ==================== 概率 -> 掩膜/彩色/ID ====================
def probs_to_masks_and_color(probs: np.ndarray, thresholds, out_size=None):
    """
    返回：bin_masks, color_mask_bgr, probs_resized, pred_bin
    """
    C, H, W = probs.shape
    
    if out_size is not None and (W != out_size[0] or H != out_size[1]): #
        up = [cv2.resize(probs[c], out_size, interpolation=cv2.INTER_LINEAR) for c in range(C)]
        probs = np.stack(up, 0)
        H, W = probs.shape[1], probs.shape[2] #

    bin_masks = []
    color = np.zeros((H, W, 3), dtype=np.uint8)
    pred_bin = np.zeros((C, H, W), dtype=np.uint8)
    for c in range(C):
        thr = float(thresholds[c])
        m = (probs[c] >= thr).astype(np.uint8) #
        pred_bin[c] = m
        bin_masks.append((m * 255).astype(np.uint8))
        color[m > 0] = PALETTE_BGR[c]
    return bin_masks, color, probs, pred_bin

def probs_to_idmask(probs: np.ndarray, pred_bin: np.ndarray) -> np.ndarray:
    """
    将多标签概率/阈值结果转为单通道 ID 掩膜：0=背景，1..5=前景
    """
    C, H, W = probs.shape
    idmask = np.zeros((H, W), np.uint8) #
    
    argmax = np.argmax(probs, axis=0).astype(np.uint8)  # H×W
    any_fg = (pred_bin.sum(axis=0) > 0)                 # 至少一类过阈值
    idmask[any_fg] = (argmax[any_fg] + 1)               # 背景=0，前景=类索引+1 #
    return idmask

def overlay_on_image(base_bgr: np.ndarray, color_mask_bgr: np.ndarray, alpha=0.5, roi_box=None):
    out = base_bgr.copy()
    if roi_box is None:
        roi_box = (0, 0, out.shape[1], out.shape[0]) #
    x0, y0, x1, y1 = roi_box
    roi = out[y0:y1, x0:x1]
    color = cv2.resize(color_mask_bgr, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    over = cv2.addWeighted(roi, 1.0, color, alpha, 0)
    out[y0:y1, x0:x1] = over
    return out

# ==================== 主推理封装函数 ====================

def run_oct_inference(
    image_path: Path, 
    model_path: Path, 
    temp_dir: Path
) -> Tuple[str, Path, Path]:
    """
    对单张图片执行 OCT 推理。
    :param image_path: 待推理的输入图片路径 (来自 FastAPI 上传)
    :param model_path: 训练好的 best_model.pth 模型路径
    :param temp_dir: 临时结果输出目录
    :return: (结果摘要字符串, ID掩膜保存路径, Overlay保存路径)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 临时配置 (固定为单次推理所需)
    INF_CFG = {
        "ckpt": str(model_path),
        "img_size": 512,
        "arch": "DeepLabV3Plus", # (使用默认值，会被 ckpt['cfg'] 覆盖)
        "encoder": "resnet34",
        "encoder_weights": "imagenet",
        "take_left_half": False, # (外部纯原图请设 False)
        "restore_to_original": True, # (掩膜还原到原图尺寸)
        "thresholds": [0.5, 0.5, 0.5, 0.5, 0.5], #
        "amp": False, # 在 FastAPI 中统一控制，此处默认 False
        "force_norm_mode": "zscore", #
    }
    
    try:
        # 载入权重与训练配置
        state, cfg_ckpt = load_checkpoint(INF_CFG["ckpt"], device) #
        state = _clean_state_dict(state)

        # 从 ckpt['cfg'] 继承关键配置
        arch    = cfg_ckpt.get("arch", INF_CFG["arch"])
        encoder = cfg_ckpt.get("encoder", INF_CFG["encoder"])
        encoder_weights = cfg_ckpt.get("encoder_weights", INF_CFG["encoder_weights"])
        img_size = int(cfg_ckpt.get("img_size", INF_CFG["img_size"]))

        # 归一化参数
        norm_mode = INF_CFG.get("force_norm_mode") or cfg_ckpt.get("norm_mode", "unit") #
        pct_low   = float(cfg_ckpt.get("pct_low", 1.0)) #
        pct_high  = float(cfg_ckpt.get("pct_high", 99.0)) #
        z_clip    = float(cfg_ckpt.get("z_clip", 3.0)) #
        stats_name = cfg_ckpt.get("norm_stats_path", "norm_stats.npz") #
        hist_name  = cfg_ckpt.get("hist_ref_path", "hist_ref.npy") #

        # 训练保存目录（用于查找统计文件，默认为模型文件所在目录）
        train_save_dir = Path(cfg_ckpt.get("save_dir", Path(INF_CFG["ckpt"]).parent))

        # 装载 zscore/hmatch 需要的统计/模板
        z_mean = z_std = hist_ref = None
        if norm_mode.lower() == "zscore":
            stats_fp = train_save_dir / stats_name
            if stats_fp.exists(): 
                d = np.load(stats_fp) 
                z_mean = d["mean"].astype(np.float32)
                z_std  = d["std"].astype(np.float32)
        elif norm_mode.lower() == "hmatch":
            hist_fp = train_save_dir / hist_name
            if hist_fp.exists(): 
                hist_ref = np.load(hist_fp).astype(np.float32) 

        model = build_model(arch, encoder, encoder_weights, classes=5).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()

        thresholds = INF_CFG["thresholds"]
        
        # --- 单图推理 ---
        img_bgr = robust_imread(image_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path.name}")

        im_tensor, orig_bgr, roi_box, roi_bgr, roi_wh = preprocess_image( #
            img_bgr=img_bgr,
            size=img_size,
            take_left_half=INF_CFG["take_left_half"],
            norm_mode=norm_mode,
            z_mean=z_mean, z_std=z_std,
            pct_low=pct_low, pct_high=pct_high, #
            z_clip=z_clip,
            hist_ref=hist_ref,
        )
        im_tensor = im_tensor.to(device)

        with torch.no_grad():
            logits = model(im_tensor) #
            probs  = torch.sigmoid(logits)[0].cpu().numpy() #

        out_size = (roi_wh[0], roi_wh[1]) if INF_CFG["restore_to_original"] else (probs.shape[2], probs.shape[1])

        # 掩膜结果
        bin_masks, color_mask_bgr, probs_resized, pred_bin = probs_to_masks_and_color(
            probs, thresholds=thresholds, out_size=out_size 
        )
        idmask = probs_to_idmask(probs_resized, pred_bin) 

        # overlay
        overlay_img = overlay_on_image(
            base_bgr=orig_bgr if INF_CFG["restore_to_original"] else roi_bgr, 
            color_mask_bgr=color_mask_bgr,
            alpha=0.5,
            roi_box=roi_box if INF_CFG["restore_to_original"] else None 
        )

        # --- 结果保存到临时目录 ---
        stem = image_path.stem
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1) ID 掩膜保存（灰度 0..5）
        id_path = temp_dir / f"{stem}_id.png"
        if not robust_imwrite(str(id_path), idmask, ext=".png", params=[cv2.IMWRITE_PNG_COMPRESSION, 3]): 
            raise RuntimeError(f"Save failed: {id_path}")
        
        # 2) Overlay 保存
        ov_path = temp_dir / f"{stem}_ov.png"
        if not robust_imwrite(str(ov_path), overlay_img, ext=".png", params=[cv2.IMWRITE_PNG_COMPRESSION, 3]): 
            raise RuntimeError(f"Save failed: {ov_path}")

        # 结果摘要
        detected_classes = [CLS_NAMES[c] for c in range(len(CLS_NAMES)) if np.sum(idmask == (c + 1)) > 0]
        summary = f"OCT 分割完成。检测到的类别: {', '.join(detected_classes) if detected_classes else '无'}"
        
        return summary, id_path, ov_path

    except Exception as e:
        raise RuntimeError(f"OCT 推理脚本执行失败: {e}")
