import torch
from PIL import Image
from utils import get_transforms, device
from model_cnn import ConvolutionalNetwork

def load_image(path):
    _, test_t = get_transforms()
    img = Image.open(path).convert("RGB")
    return test_t(img).unsqueeze(0)

def main(image, model_path="CNNModel.pt", class_names=None):
    d = device()
    model = ConvolutionalNetwork().to(d)
    model.load_state_dict(torch.load(model_path, map_location=d))
    model.eval()

    x = load_image(image).to(d)
    with torch.no_grad():
        y = model(x).argmax(dim=1).item()

    if class_names is None:
        class_names = ["bad", "good"]
    print(f"Prediction: {y} ({class_names[y]})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model_path", default="CNNModel.pt")
    ap.add_argument("--class_names", nargs="*", default=None)
    args = ap.parse_args()
    main(**vars(args))
