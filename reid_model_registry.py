"""
Re-ID Model Registry
Supports multiple SOTA Re-ID models for person and vehicle re-identification
"""

import torch
import os
import sys
import urllib.request
from typing import Optional, Tuple

# Check if torchreid is available
try:
    from torchreid import models, utils
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("⚠️ Warning: torchreid not installed")

# Check if fast-reid is available
FASTREID_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast-reid")
if os.path.exists(FASTREID_ROOT):
    sys.path.append(FASTREID_ROOT)
    try:
        from fastreid.config import get_cfg
        from fastreid.modeling import build_model
        from fastreid.utils.checkpoint import Checkpointer
        FASTREID_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ Warning: fast-reid found but import failed: {e}")
        FASTREID_AVAILABLE = False
else:
    FASTREID_AVAILABLE = False


# ============================================================================
# FastReID Wrapper
# ============================================================================
class FastReIDWrapper(torch.nn.Module):
    """Wrapper to make FastReID models compatible with TorchReID interface"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # FastReID expects dict with "images" key
        # x is (B, C, H, W) tensor
        if self.model.training:
            return self.model({"images": x})
        else:
            # Inference: returns features
            return self.model({"images": x})


# ============================================================================
# Model Registry: Person Re-ID Models
# ============================================================================

PERSON_REID_MODELS = {
    # OSNet variants (TorchReID built-in)
    "osnet_x1_0": {
        "name": "OSNet x1.0",
        "description": "Original OSNet (TorchReID)",
        "num_classes": 0,  # Pretrained
        "pretrained_name": "osnet_x1_0",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953"
    },
    "osnet_x0_75": {
        "name": "OSNet x0.75",
        "description": "Lightweight OSNet (faster)",
        "num_classes": 0,
        "pretrained_name": "osnet_x0_75",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953"
    },
    "osnet_x0_5": {
        "name": "OSNet x0.5",
        "description": "Most lightweight OSNet (fastest)",
        "num_classes": 0,
        "pretrained_name": "osnet_x0_5",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953"
    },
    "osnet_ibn_x1_0": {
        "name": "OSNet-IBN x1.0",
        "description": "OSNet with Instance-Batch Normalization",
        "num_classes": 0,
        "pretrained_name": "osnet_ibn_x1_0",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953"
    },

    # Market-1501 fine-tuned (external weights)
    "osnet_market1501": {
        "name": "OSNet Market-1501",
        "description": "OSNet fine-tuned on Market-1501",
        "num_classes": 751,
        "pretrained_name": None,
        "local_path": "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953",
        "note": "Best for person Re-ID"
    },

    # ResNet variants (TorchReID built-in)
    "resnet50_fc512": {
        "name": "ResNet50-FC512",
        "description": "ResNet50 with 512-dim features",
        "num_classes": 0,
        "pretrained_name": "resnet50_fc512",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1512.03385"
    },

    # FastReID Models
    "fastreid_sbs_r50": {
        "name": "FastReID SBS-R50",
        "description": "Strong Baseline with ResNet50 (FastReID)",
        "num_classes": 1501,
        "config_file": "fast-reid/configs/Market1501/sbs_R50.yml",
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth",
        "local_path": "market_sbs_R50.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/2006.02631",
        "note": "⭐ SOTA Performance"
    },
    "fastreid_sbs_r101": {
        "name": "FastReID SBS-R101-IBN",
        "description": "Strong Baseline with ResNet101-IBN (FastReID)",
        "num_classes": 1501,
        "config_file": "fast-reid/configs/Market1501/sbs_R101-ibn.yml",
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R101-ibn.pth",
        "local_path": "market_sbs_R101-ibn.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/2006.02631",
        "note": "⭐ Best Accuracy (Heavy)"
    },
    "fastreid_agw_r101": {
        "name": "FastReID AGW-R101-IBN",
        "description": "Attention-Guided Weighting with ResNet101-IBN (FastReID)",
        "num_classes": 1501,
        "config_file": "fast-reid/configs/Market1501/AGW_R101-ibn.yml",
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R101-ibn.pth",
        "local_path": "market_agw_R101-ibn.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/2006.02631",
        "note": "⭐ SOTA Performance (95.5% Rank@1)"
    },
    "fastreid_mgn_r50": {
        "name": "FastReID MGN-R50-IBN",
        "description": "Multiple Granularity Network with ResNet50-IBN (FastReID)",
        "num_classes": 1501,
        "config_file": "fast-reid/configs/Market1501/mgn_R50-ibn.yml",
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_mgn_R50-ibn.pth",
        "local_path": "market_mgn_R50-ibn.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/1804.01438",
        "note": "⭐ Multi-Granularity Features"
    },
}


# ============================================================================
# Model Registry: Vehicle Re-ID Models
# ============================================================================

VEHICLE_REID_MODELS = {
    # OSNet variants for vehicles
    "osnet_x1_0": {
        "name": "OSNet x1.0 (Pretrained)",
        "description": "OSNet pretrained on ImageNet",
        "num_classes": 0,
        "pretrained_name": "osnet_x1_0",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953"
    },

    # VeRi-776 fine-tuned (if available)
    "osnet_veri776": {
        "name": "OSNet VeRi-776",
        "description": "OSNet fine-tuned on VeRi-776",
        "num_classes": 776,
        "pretrained_name": None,
        "local_path": "osnet_x1_0_veri_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1905.00953",
        "note": "Best for vehicle Re-ID (if available)"
    },

    # ResNet for vehicles
    "resnet50_fc512": {
        "name": "ResNet50-FC512",
        "description": "ResNet50 baseline for vehicles",
        "num_classes": 0,
        "pretrained_name": "resnet50_fc512",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1512.03385"
    },

    # FastReID Vehicle Models
    "fastreid_veri_sbs_r50": {
        "name": "FastReID VeRi SBS-R50",
        "description": "Strong Baseline on VeRi-776 (FastReID)",
        "num_classes": 776,
        "config_file": "fast-reid/configs/VeRi/sbs_R50-ibn.yml",
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth",
        "local_path": "veri_sbs_R50-ibn.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/2006.02631",
        "note": "⭐ SOTA for Vehicles (VeRi)"
    },
    "fastreid_vehicleid_bot_r50": {
        "name": "FastReID VehicleID BoT-R50",
        "description": "Bag of Tricks on VehicleID (FastReID)",
        "num_classes": 13164, # Large scale
        "config_file": "fast-reid/configs/VehicleID/bagtricks_R50-ibn.yml",
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth",
        "local_path": "vehicleid_bot_R50-ibn.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/2006.02631",
        "note": "⭐ SOTA for Vehicles (Large Scale)"
    },
    "fastreid_veri_agw_r50": {
        "name": "FastReID VeRi AGW-R50",
        "description": "Attention-Guided Weighting on VeRi-776 (FastReID)",
        "num_classes": 776,
        "config_file": "fast-reid/configs/VeRi/sbs_R50-ibn.yml",  # Using SBS config as AGW config may not exist
        "url": "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth",
        "local_path": "veri_sbs_R50-ibn.pth",
        "framework": "fastreid",
        "paper": "https://arxiv.org/abs/2006.02631",
        "note": "⭐ Alternative SOTA for Vehicles"
    },
}


# ============================================================================
# Model Loading Functions
# ============================================================================

def download_file(url, dest_path):
    """Download file with progress bar"""
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("✓ Download complete")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def load_fastreid_model(model_info, device):
    """Load FastReID model"""
    if not FASTREID_AVAILABLE:
        print("⚠️ FastReID not available")
        return None

    config_file = model_info.get("config_file")
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        return None

    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.BACKBONE.PRETRAIN = False  # Don't download ImageNet weights if loading checkpoint
    cfg.MODEL.DEVICE = device
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load weights
    weights_path = model_info.get("local_path")
    if not os.path.exists(weights_path):
        url = model_info.get("url")
        if url:
            print(f"   Weights not found locally. Downloading from {url}...")
            if not download_file(url, weights_path):
                return None
        else:
            print(f"✗ Weights not found and no URL provided: {weights_path}")
            return None
            
    print(f"   Loading weights from: {weights_path}")
    Checkpointer(model).load(weights_path)
    
    return FastReIDWrapper(model)

def load_reid_model(
    model_key: str,
    model_type: str = "person",
    device: str = "cuda",
    local_weights_path: Optional[str] = None
) -> Tuple[Optional[torch.nn.Module], dict]:
    """
    Load a Re-ID model from the registry

    Args:
        model_key: Model key from registry (e.g., "osnet_x1_0", "osnet_market1501")
        model_type: "person" or "vehicle"
        device: "cuda" or "cpu"
        local_weights_path: Optional path to local weights (overrides registry)

    Returns:
        Tuple of (model, model_info)
    """
    # Select registry
    registry = PERSON_REID_MODELS if model_type == "person" else VEHICLE_REID_MODELS

    if model_key not in registry:
        available = list(registry.keys())
        raise ValueError(f"Model '{model_key}' not found. Available: {available}")

    model_info = registry[model_key]
    print(f"\n📊 Loading {model_type.upper()} Re-ID Model: {model_info['name']}")
    print(f"   Description: {model_info['description']}")
    
    framework = model_info.get("framework", "torchreid")
    
    # Handle FastReID
    if framework == "fastreid":
        try:
            model = load_fastreid_model(model_info, device)
            if model:
                print(f"   ✓ FastReID model ready on {device}")
                return model, model_info
            else:
                print("   ✗ Failed to load FastReID model")
                return None, model_info
        except Exception as e:
            print(f"   ✗ Error loading FastReID model: {e}")
            import traceback
            traceback.print_exc()
            return None, model_info

    # Handle TorchReID
    if not TORCHREID_AVAILABLE:
        print("⚠️ torchreid not available")
        return None, {}

    try:
        # Determine model architecture
        arch_name = model_key.replace("_market1501", "").replace("_veri776", "")

        # Build model
        num_classes = model_info['num_classes']
        model = models.build_model(name=arch_name, num_classes=num_classes)
        model = model.to(device)

        # Load weights
        weights_path = local_weights_path or model_info.get('local_path')

        if weights_path and os.path.exists(weights_path):
            # Load local weights
            print(f"   Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"   ✓ Loaded fine-tuned weights")

        elif model_info.get('pretrained_name'):
            # Load pretrained weights from torchreid
            print(f"   Loading pretrained weights: {model_info['pretrained_name']}")
            utils.load_pretrained_weights(model, model_info['pretrained_name'])
            print(f"   ✓ Loaded pretrained weights")

        else:
            print(f"   ⚠️ No weights found, using random initialization")

        # Remove classifier for feature extraction
        if hasattr(model, 'classifier') and num_classes > 0:
            model.classifier = None

        model.eval()

        print(f"   ✓ Model ready on {device}")
        if 'note' in model_info:
            print(f"   💡 Note: {model_info['note']}")

        return model, model_info

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return None, model_info


def list_available_models(model_type: str = "person") -> None:
    """
    Print available Re-ID models

    Args:
        model_type: "person" or "vehicle"
    """
    registry = PERSON_REID_MODELS if model_type == "person" else VEHICLE_REID_MODELS

    print(f"\n{'='*80}")
    print(f"Available {model_type.upper()} Re-ID Models")
    print(f"{'='*80}")

    for idx, (key, info) in enumerate(registry.items(), 1):
        print(f"\n{idx}. Key: '{key}'")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Framework: {info.get('framework', 'torchreid')}")

        if 'local_path' in info:
            path = info['local_path']
            exists = "✓" if os.path.exists(path) else "✗ (not found)"
            print(f"   Local weights: {path} {exists}")
        elif info.get('pretrained_name'):
            print(f"   Pretrained: {info['pretrained_name']} (auto-download)")

        if 'note' in info:
            print(f"   💡 {info['note']}")

    print(f"\n{'='*80}\n")


def get_model_info(model_key: str, model_type: str = "person") -> dict:
    """Get information about a model"""
    registry = PERSON_REID_MODELS if model_type == "person" else VEHICLE_REID_MODELS
    return registry.get(model_key, {})


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Re-ID Model Registry")
    print("=" * 80)

    # List available models
    list_available_models("person")
    list_available_models("vehicle")

    # Test loading a model
    if TORCHREID_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("\nTesting model loading...")
        # Test FastReID if available
        if FASTREID_AVAILABLE:
            print("\nTesting FastReID loading...")
            model, info = load_reid_model("fastreid_sbs_r50", "person", device)
        else:
            model, info = load_reid_model("osnet_x1_0", "person", device)

        if model:
            print(f"\n✓ Successfully loaded: {info['name']}")
            print(f"  Model type: {type(model)}")
