"""
Wrist Image Classification - Inference Script
==============================================
Standalone script for loading the trained model and making predictions.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys
from pathlib import Path


class WristClassifier(nn.Module):
    """MobileNetV2-based classifier for wrist images."""
    
    def __init__(self, num_classes=3, pretrained=False):
        super(WristClassifier, self).__init__()
        
        # Load MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of input features
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, device='cpu'):
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = WristClassifier(num_classes=checkpoint['num_classes'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict(image_path, model_path='model.pth', device=None, verbose=True):
    """
    Predict the class of a wrist image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model checkpoint
        device: Device to use ('cpu' or 'cuda'). Auto-detect if None
        verbose: Print detailed results
        
    Returns:
        Dictionary with prediction results:
            - predicted_class: Predicted class name
            - confidence: Confidence score (0-1)
            - probabilities: Dictionary of all class probabilities
    """
    # Auto-detect device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Validate image path
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    
    # Prepare image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(checkpoint['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint['normalization']['mean'],
            std=checkpoint['normalization']['std']
        )
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Handle both dict with int and str keys
        idx_to_class = checkpoint['idx_to_class']
        if isinstance(list(idx_to_class.keys())[0], str):
            predicted_class = idx_to_class[str(predicted_idx.item())]
        else:
            predicted_class = idx_to_class[predicted_idx.item()]
    
    # Get all class probabilities
    class_probs = {}
    for i in range(len(idx_to_class)):
        key = str(i) if isinstance(list(idx_to_class.keys())[0], str) else i
        class_probs[idx_to_class[key]] = probabilities[0][i].item()
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence.item(),
        'probabilities': class_probs
    }
    
    # Print results if verbose
    if verbose:
        print(f"\n{'='*60}")
        print(f"Image: {Path(image_path).name}")
        print(f"{'='*60}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll Class Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 40)
            print(f"  {cls:10s} {prob:.2%} {bar}")
        print(f"{'='*60}\n")
    
    return result


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(
        description='Wrist Image Classification - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py image.jpg
  python inference.py image.jpg --model custom_model.pth
  python inference.py image.jpg --device cuda --quiet
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the wrist image file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='model.pth',
        help='Path to the model checkpoint (default: model.pth)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output, only print predicted class'
    )
    
    args = parser.parse_args()
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    try:
        # Run prediction
        result = predict(
            args.image_path,
            model_path=args.model,
            device=device,
            verbose=not args.quiet
        )
        
        # Print minimal output if quiet mode
        if args.quiet:
            print(result['predicted_class'])
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
