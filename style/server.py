from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from flask import abort, Flask, jsonify, request, Response
from flask_cors import CORS
import torch
import torchvision.transforms as transforms

from transformer_net import TransformerNet
import utils

app = Flask(__name__)
CORS(app)


def load_model(style: str) -> TransformerNet:
    model = TransformerNet()
    state_dict = torch.load(str(Path(__file__).parent
                                / "saved_models"
                                / ("%s.pth" % style)))
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    model.load_state_dict(state_dict)
    return model


models = {
    "mosaic": load_model("mosaic"),
    "udnie": load_model("udnie")
}


@app.route('/', methods=["POST"])
def stylize() -> Response:
    body = request.get_json()
    img_path = body["img_path"]
    tone = body["tone"]
    if tone not in ["happy", "sad"]:
        abort(400)
    model = models["mosaic"] if tone == "happy" else models["udnie"]
    content_image = utils.load_image(img_path)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    with torch.no_grad():
        output = model(content_image)
    with NamedTemporaryFile(prefix="style-%s-" % tone,
                            suffix=".jpg",
                            delete=False) as fh:
        utils.save_image(fh.name, output[0])
    return jsonify({"stylized_img_path": fh.name})
