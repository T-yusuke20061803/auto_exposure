import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_p = 0.5, conv_dropout_p = 0.2):#大きすぎると過学習、小さすぎると学習不足の影響があるため、調整が必要
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
                     down_sampling_layer=nn.Conv2d, dropout_p=0.3):
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
                     down_sampling_layer=nn.Conv2d, dropout_p=0.3):
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
                 down_sampling_layer=nn.Conv2d, dropout_p=0.3):
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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 過学習するようであれば、レイヤー数とニューロン数を小さくする
        self.linear = nn.Sequential(
            nn.Linear(512 * block.expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),

            nn.Linear(256* block.expansion, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p * 0.7),

            nn.Linear(128* block.expansion, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p * 0.5),

            nn.Linear(64* block.expansion, 32),
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


class RegressionEfficientNet(nn.Module):
    """
    EfficientNet-B0をベースに、露出値回帰用にカスタマイズした軽量モデル
    （過学習抑制と汎化性能向上を重視）
    """
    def __init__(self, version='b0', out_features=1, freeze_base=True, unfreeze_layers=2, dropout_p =0.5):#versonでモデルの種類を指定 
        super().__init__() 
        if version.lower() == 'b0': 
            weights = models.EfficientNet_B0_Weights.DEFAULT 
            self.effnet = models.efficientnet_b0(weights=weights) 
        elif version.lower() == 'b1': 
            weights = models.EfficientNet_B1_Weights.DEFAULT 
            self.effnet = models.efficientnet_b1(weights=weights) 
        elif version.lower() == 'b2': 
            weights = models.EfficientNet_B2_Weights.DEFAULT 
            self.effnet = models.efficientnet_b2(weights=weights) 
        elif version.lower() == 'b3': 
            weights = models.EfficientNet_B3_Weights.DEFAULT 
            self.effnet = models.efficientnet_b3(weights=weights) 
        elif version.lower() == 'b4': 
            weights = models.EfficientNet_B4_Weights.DEFAULT 
            self.effnet = models.efficientnet_b4(weights=weights) 
        elif version.lower() == 'b5': 
            weights = models.EfficientNet_B5_Weights.DEFAULT 
            self.effnet = models.efficientnet_b5(weights=weights) 
        elif version.lower() == 'b6': 
            weights = models.EfficientNet_B6_Weights.DEFAULT 
            self.effnet = models.efficientnet_b6(weights=weights) 
        elif version.lower() == 'b7': 
            weights = models.EfficientNet_B7_Weights.DEFAULT 
            self.effnet = models.efficientnet_b7(weights=weights) 
        else: 
            raise ValueError(f"Unsupported EfficientNet version: {version}")
        
        # --- 特徴抽出部の凍結 ---
        if freeze_base:
            for param in self.effnet.features.parameters():
                param.requires_grad = False

        # unfreeze_layers の指定分だけ後ろから解凍
        if unfreeze_layers > 0:
            for block in range(-unfreeze_layers, 0):
                for param in self.effnet.features[block].parameters():
                    param.requires_grad = True

        # --- 分類層の再構築（Dropout強化・BatchNorm追加） ---
        num_ftrs = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.7),
            nn.Linear(128, 64),
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
    def __init__(self, out_features=1, freeze_base=True, unfreeze_layers=0, dropout_p=0.5):
        super().__init__()
        
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
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
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.7),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p*0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)
        )

    def forward(self, x):
        return self.mobilenet(x)