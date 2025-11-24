"""
Re-ID Model Registry
Supports multiple SOTA Re-ID models for person and vehicle re-identification
"""

import torch
import os
from typing import Optional, Tuple

# Check if torchreid is available
try:
    from torchreid import models, utils
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("⚠️ Warning: torchreid not installed")


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

    # MobileNet variants (lightweight)
    "mlfn": {
        "name": "MLFN",
        "description": "Multi-Level Factorisation Net (lightweight)",
        "num_classes": 0,
        "pretrained_name": "mlfn",
        "framework": "torchreid",
        "paper": "https://arxiv.org/abs/1803.09132"
    },

    # SOTA Models (External - require separate installation)
    # Note: These models require additional setup beyond torchreid
    "solider": {
        "name": "SOLIDER",
        "description": "Semantic-Controllable Self-Supervised Learning (SOTA 2023)",
        "num_classes": 0,
        "pretrained_name": None,
        "local_path": "solider_pretrained.pth",
        "framework": "custom",  # Requires SOLIDER-REID repo
        "paper": "https://arxiv.org/abs/2303.16317",
        "github": "https://github.com/tinyvision/SOLIDER-REID",
        "note": "⭐ SOTA performance, used by NVIDIA. Requires separate installation."
    },

    "personvit": {
        "name": "PersonViT",
        "description": "Large-scale Self-supervised ViT for Person Re-ID (2024)",
        "num_classes": 0,
        "pretrained_name": None,
        "local_path": "personvit_pretrained.pth",
        "framework": "custom",  # Requires PersonViT repo
        "paper": "https://arxiv.org/abs/2408.05398",
        "github": "https://github.com/hustvl/PersonViT",
        "note": "⭐ Latest SOTA (Aug 2024). Requires separate installation."
    },

    "clip_reid": {
        "name": "CLIP-ReID",
        "description": "Vision-Language Model for Re-ID (AAAI 2023)",
        "num_classes": 0,
        "pretrained_name": None,
        "local_path": "clip_reid_pretrained.pth",
        "framework": "custom",  # Requires CLIP-ReID repo
        "paper": "https://arxiv.org/abs/2211.13977",
        "github": "https://github.com/Syliz517/CLIP-ReID",
        "note": "Uses vision-language pre-training. Requires separate installation."
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
}


# ============================================================================
# Model Loading Functions
# ============================================================================

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
    if not TORCHREID_AVAILABLE:
        print("⚠️ torchreid not available")
        return None, {}

    # Select registry
    registry = PERSON_REID_MODELS if model_type == "person" else VEHICLE_REID_MODELS

    if model_key not in registry:
        available = list(registry.keys())
        raise ValueError(f"Model '{model_key}' not found. Available: {available}")

    model_info = registry[model_key]
    print(f"\n📊 Loading {model_type.upper()} Re-ID Model: {model_info['name']}")
    print(f"   Description: {model_info['description']}")

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
        model, info = load_reid_model("osnet_x1_0", "person", device)

        if model:
            print(f"\n✓ Successfully loaded: {info['name']}")
            print(f"  Model type: {type(model)}")
