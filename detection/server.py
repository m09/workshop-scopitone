from pathlib import Path

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import torch
from torch.nn import Linear
from torchvision.models import resnet50
import torchvision.transforms as transforms

import utils

app = Flask(__name__)
CORS(app)


def load_model() -> resnet50:
    model = resnet50()
    model.fc = Linear(in_features=2048, out_features=2, bias=True)
    state_dict = torch.load(str(Path(__file__).parent.parent / "model.pth"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()


@app.route('/', methods=["POST"])
def stylize() -> Response:
    body = request.get_json()
    img_path = body["img_path"]
    content_image = utils.load_image(img_path)
    content_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    with torch.no_grad():
        output = model(content_image)
        prediction = output.argmax(dim=1)[0].item()
    return jsonify({"predicted": prediction})
