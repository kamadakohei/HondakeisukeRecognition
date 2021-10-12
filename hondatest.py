from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import os
import shutil
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["本田圭佑", "じゅんいちダビットソン"]
n_class = len(labels)
img_size = 256
#n_result = 2  # 上位2つの結果を表示

# 転移学習用のResnet18をロード
net = models.resnet18(pretrained=True)

# すべてのパラメータを微分対象外にする
for p in net.parameters():
    p.requires_grad=False

# 全結合層、ドロップアウト層を加える
net.fc = nn.Linear(512, 200)
net.drop = nn.Dropout(0.25)
net.fc2 = nn.Linear(200, 2)
print(net)

net.load_state_dict(torch.load("honda_cnn.pth", map_location=torch.device("cpu")))  #CPU対応
net.eval() # 評価モード


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))

        normalize = transforms.Normalize(
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均値を0、標準偏差を1に
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([to_tensor, normalize])

        x = transform(image)
        x = x.reshape(1, 3, img_size, img_size)

        print(x.size())
        y = net(x)
        print(y.size())
        result = ""
        label = labels[y.argmax().item()]
        result += "<p>" + str(round(F.softmax(y).max().item()*100, 1)) + \
            "%の確率で" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))


if __name__ == "__main__":
    
    app.run(debug=True)
