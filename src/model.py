import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_p = 0.4, conv_dropout_p = 0.4):#大きすぎると過学習、小さすぎると学習不足の影響があるため、調整が必要
        """DNNの層を定義
        """
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            # Block 1
            nn.LazyConv2d(out_channels=32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.LazyConv2d(out_channels=128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.LazyConv2d(out_channels=512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            nn.LazyLinear(out_features=num_classes)
        )
        
    
    def forward(self, x):
        """DNNの入力から出力までの計算
        Args:
            x: torch.Tensor whose size of
               (batch size, # of channels, # of freq. bins, # of time frames)
        Return:
            y: torch.Tensor whose size of
               (batch size, # of classes)
        """
        #変更前
        #x = self.net(x)
        #x = x.view(x.size(0),-1)
        #x = self.dropout(x)
        #y = self.classifier(x.view(x.size(0), -1))
        #return y
        #変更後
        # 畳み込み部分で特徴抽出
        features = self.net(x)
        # 最終判断部分に渡す（中で自動的にFlattenされる）
        return self.classifier(features)


class ResNet(nn.Module):
    """ResNet18, 34, 50, 101, and 152

    >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    >>> net = ResNet('ResNet18').to(device)
    """
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1,
                     down_sampling_layer=nn.Conv2d, dropout_p=0.4):
            super(ResNet.BasicBlock, self).__init__()
            if stride != 1:
                self.conv1 = down_sampling_layer(
                    in_planes, planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)
            else:
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False)

            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

            #ドロップアウトを追加して過学習を抑制(10/08)
            self.dropout = nn.Dropout2d(p=dropout_p)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    down_sampling_layer(
                        in_planes, self.expansion*planes, kernel_size=1,
                        stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1,
                     down_sampling_layer=nn.Conv2d, dropout_p=0.4):
            super(ResNet.Bottleneck, self).__init__()
            
            self.conv1 = nn.Conv2d(in_planes, planes,
                                   kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            if stride != 1:
                self.conv2 = down_sampling_layer(
                    planes, planes, kernel_size=3,
                    stride=stride, padding=1, bias=False)
            else:
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                       stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                                   kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

            self.dropout = nn.Dropout2d(p=dropout_p)

            self.shortcut = nn.Sequential()
            if stride != 1:
                self.shortcut = nn.Sequential(
                    down_sampling_layer(
                        in_planes, self.expansion*planes, kernel_size=1,
                        stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            elif in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.dropout(out)
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    def __init__(self, resnet_name, num_classes=1,
                 down_sampling_layer=nn.Conv2d, dropout_p=0.4):
        super(ResNet, self).__init__()
        if resnet_name == "ResNet18":
            block = ResNet.BasicBlock
            num_blocks = [2, 2, 2, 2]
        elif resnet_name == "ResNet34":
            block = ResNet.BasicBlock
            num_blocks = [3, 4, 6, 3]
        elif resnet_name == "ResNet50":
            block = ResNet.Bottleneck
            num_blocks = [3, 4, 6, 3]
        elif resnet_name == "ResNet101":
            block = ResNet.Bottleneck
            num_blocks = [3, 4, 23, 3]
        elif resnet_name == "ResNet152":
            block = ResNet.Bottleneck
            num_blocks = [3, 8, 36, 3]
        else:
            raise NotImplementedError()

        self.in_planes = 64
        self.down_sampling_layer = down_sampling_layer
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 過学習するようであれば、レイヤー数とニューロン数を小さくする
        self.linear = nn.Sequential(
            nn.Linear(512* block.expansion, 512),#大きく減らしすぎた可能性がある10/24
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p * 0.7),

            
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p * 0.3),

            nn.Linear(32, num_classes)
        )


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                down_sampling_layer=self.down_sampling_layer,
                                dropout_p=self.dropout_p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)) # 常に 1x1 に要約する
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetRegression(nn.Module):
    """
    torchvision の事前学習済み ResNet をベースとし、
    カスタム回帰ヘッドを持つモデル
    """
    def __init__(self, resnet_name="ResNet34", out_features=1, freeze_base=True, dropout_p=0.3, unfreeze_layers=0):
        super().__init__()

        # 事前学習済みのResNetを読み込む
        if resnet_name == "ResNet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
            num_ftrs = 512 # ResNet18/34 の fc 入力
        elif resnet_name == "ResNet34":
            weights = models.ResNet34_Weights.DEFAULT
            self.resnet = models.resnet34(weights=weights)
            num_ftrs = 512 # ResNet18/34 の fc 入力
        elif resnet_name == "ResNet50":
            weights = models.ResNet50_Weights.DEFAULT
            self.resnet = models.resnet50(weights=weights)
            num_ftrs = 2048 # ResNet50/101/152 の fc 入力
        # (ResNet101なども同様に追加可能)
        else:
            raise ValueError(f"未対応のResNet名: {resnet_name}")

        # ベース層を凍結 (転移学習の基本)

        if freeze_base:
            # まず、全てのパラメータを凍結
            for param in self.resnet.parameters():
                param.requires_grad = False
            
            # 指定された層数だけ解凍 (EfficientNet と同じロジック)
            if unfreeze_layers > 0:
                # 例: unfreeze_layers=1 なら layer4 を解凍
                if hasattr(self.resnet, 'layer4'):
                    for param in self.resnet.layer4.parameters():
                        param.requires_grad = True
            if unfreeze_layers > 1:
                # 例: unfreeze_layers=2 なら layer3 も解凍
                if hasattr(self.resnet, 'layer3'):
                    for param in self.resnet.layer3.parameters():
                        param.requires_grad = True

        # --- 3. 最後の層(fc)を、カスタム回帰ヘッドに置き換える ---
        # (先生のコードの self.linear の構造を再現)
        self.resnet.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_ftrs, out_features)
        )

    def forward(self, x):
        # torchvision の ResNet をそのまま実行
        return self.resnet(x)

class RegressionEfficientNet(nn.Module):
    """
    EfficientNet-B0をベースに、露出値回帰用にカスタマイズした軽量モデル
    （過学習抑制と汎化性能向上を重視）
    """
    def __init__(self, version='b0', out_features=1, freeze_base=True, unfreeze_layers=2, dropout_p =0.4, pretrained=True):#versonでモデルの種類を指定 :pretrained=True:事前学習有り、pretrained=False:事前学習無し
        super().__init__() 
        version = version.lower()
        valid_versions = [f"b{i}" for i in range(8)]
        if version not in valid_versions:
            raise ValueError(f"Unsupported EfficientNet version: {version}. Choose from {valid_versions}")

        # EfficientNetのモデルと重みを自動的に選択 
        model_fn = getattr(models, f"efficientnet_{version}")
        weight_enum = getattr(models, f"EfficientNet_{version.upper()}_Weights")
        weights = weight_enum.DEFAULT if pretrained else None

        # モデル構築
        self.effnet = model_fn(weights=weights)
        
        # 特徴抽出部の凍結
        if freeze_base:
            for param in self.effnet.features.parameters():
                param.requires_grad = False

        # unfreeze_layers の指定分だけ後ろから解凍
        if unfreeze_layers > 0:
            for block in range(-unfreeze_layers, 0):
                for param in self.effnet.features[block].parameters():
                    param.requires_grad = True

        # 分類層の再構築（Dropout強化・BatchNorm追加）
        num_ftrs = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.7),

            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.3),
            
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)
        )

    def forward(self, x):
        return self.effnet(x)


class RegressionMobileNet(nn.Module):
    """
    MobileNetV2をベースにした軽量回帰モデル
    小型かつ高汎化（過学習抑制・正則化強化）
    """
    def __init__(self, out_features=1, freeze_base=True, unfreeze_layers=0, dropout_p=0.4, pretrained=True):
        super().__init__()
        
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.mobilenet = models.mobilenet_v3_large(weights=weights)
        
        # --- 特徴抽出層を凍結 ---
        if freeze_base:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False

        # unfreeze_layers分だけ後方からアンフリーズ
        if unfreeze_layers > 0:
            for block in range(-unfreeze_layers, 0):
                for param in self.mobilenet.features[block].parameters():
                    param.requires_grad = True

        # --- classifierの再構築 ---
        num_ftrs = self.mobilenet.classifier[0].in_features
        self.effnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.7),

            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.3),
            
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)
        )
    def forward(self, x):
        return self.mobilenet(x)