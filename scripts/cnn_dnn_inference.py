# scripts/cnn_dnn_inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional
import timm 
import numpy as np

# ===============================================================
# 1. 模型子结构定义 (ConvNeXtEncoder, MPIDNN)
#    --- 基于用户提供的完整模型定义 ---
# ===============================================================

# 1.1 ConvNeXtEncoder (图像分支)
class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        out_dim: int = 32,
        pretrained: bool = True,
        ckpt_path: Optional[Path] = None, 
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=3,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features # ConvNeXt-Tiny 默认为 768
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)     # [B, 768]
        out  = self.fc(feat)        # [B, 32]
        out  = F.relu(out, inplace=True)
        return out

# 1.2 MPIDNN (临床/MPI 分支)
class MPIDNN(nn.Module):
    # in_dim=2, hidden_dim=32, out_dim=16，这些参数来自您的训练脚本
    def __init__(self, in_dim: int = 2, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 1.3 MultiModalNet (融合模型主体)
class MultiModalNet(nn.Module):
    def __init__(
        self,
        img_dim: int = 32,
        mpi_dim: int = 16,
        fused_dim: int = 64,
        num_classes: int = 2, # 确定为 2 类
        dropout: float = 0.5,
        backbone_name: str = "convnext_tiny",
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.cnn = ConvNeXtEncoder(
            model_name=backbone_name,
            out_dim=img_dim,
            pretrained=pretrained_backbone,
            # 部署时不需要 CONVNEXT_CKPT 路径
        )
        self.dnn = MPIDNN(in_dim=2, hidden_dim=32, out_dim=mpi_dim)

        # 融合维度：32 + 16 = 48
        self.fuse_mlp = nn.Sequential(
            nn.Linear(img_dim + mpi_dim, fused_dim), # 48 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, num_classes)        # 64 -> 2 (最终分类)
        )

    def forward(self, img: torch.Tensor, mpi: torch.Tensor) -> torch.Tensor:
        vi = self.cnn(img)
        vt = self.dnn(mpi)
        v  = torch.cat([vi, vt], dim=1) # [B, 48]
        logits = self.fuse_mlp(v)
        return logits
# ===============================================================


# 2. 预处理与标签
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ConvNeXt 默认输入尺寸
    transforms.ToTensor(),
    # ImageNet 标准归一化，请确保您的训练流程一致
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 类别标签 (二分类)
# -------------------------------------------------------------
# *** 警告：请务必替换为您模型训练时的实际类别名称和顺序！ ***
# -------------------------------------------------------------
CLASS_LABELS = ["无响应", "有响应"] 

# **硬编码的 MPI 默认输入** (2 维特征，例如：[归一化年龄, 性别])
# 由于API不接收MPI输入，此处使用默认值。在生产环境中应由用户提供。
DUMMY_MPI_INPUT = torch.tensor([[0.5, 0.0]], dtype=torch.float32) 

def run_cnn_dnn_inference(image_path: Path, model_path: Path) -> str:
    """
    加载 MultiModalNet（cnn_dnn_best_703.pt）并对给定图片和硬编码的MPI数据进行推理。
    """
    if not model_path.exists():
        return f"错误：模型文件 {model_path.name} 不存在。"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 尝试加载模型
    try:
        # 实例化模型，使用训练时的默认参数
        model = MultiModalNet(
            num_classes=len(CLASS_LABELS), 
            img_dim=32, mpi_dim=16, fused_dim=64,
            backbone_name="convnext_tiny",
            dropout=0.5
        )
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.to(device)
        model.eval()
    except Exception as e:
        return (f"错误：无法加载或初始化模型 {model_path.name}。请检查 MultiModalNet 定义是否与训练时一致。原始错误: {e}")

    # 加载和预处理图片
    try:
        image = Image.open(image_path).convert("RGB")
        input_img_tensor = transform(image)
        input_img_batch = input_img_tensor.unsqueeze(0).to(device)
        
        # 准备 MPI 批次输入 (硬编码默认值)
        input_mpi_batch = DUMMY_MPI_INPUT.to(device)
        
    except Exception as e:
        return f"错误：无法加载或预处理图片/MPI数据。原因: {e}"

    # 执行推理
    with torch.no_grad():
        # **模型需要两个输入：图像和 MPI**
        logits = model(input_img_batch, input_mpi_batch)
        
    # 处理输出：使用 Softmax 后的概率
    probs = F.softmax(logits, dim=1) 
    
    # 获取最高概率的类别和概率值
    max_prob, predicted_index = torch.max(probs, 1)
    
    # 格式化结果
    pred_class = CLASS_LABELS[predicted_index.item()]
    confidence = max_prob.item()

    result_str = (
        f"**模型类型**: 多模态二分类\n"
        f"**硬编码 MPI 输入**: {DUMMY_MPI_INPUT.tolist()} (请注意此值为默认值)\n"
        f"**预测类别**: {pred_class} (置信度: {confidence:.4f})\n"
        f"**所有类别概率**: {probs.squeeze().tolist()}"
    )

    return result_str
