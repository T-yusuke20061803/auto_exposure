import time
import os
from abc import ABC
from abc import abstractmethod
import typing

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch import autograd
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler #AMP対応


# Utility: 平均値を取るクラス
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


# 抽象トレーナークラス
class ABCTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    @abstractmethod
    def eval(self):
        raise NotImplementedError()

    @abstractmethod
    def extend(self):
        raise NotImplementedError()

#トレーナー本体
class Trainer(ABCTrainer):
    def __init__(self,
                 net,
                 optimizer,
                 criterion,
                 dataloader,
                 cfg,
                 scheduler=None,
                 extensions=None,
                 evaluators=None,
                 init_epoch=0,
                 device='cpu'):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.cfg = cfg
        self.scheduler = scheduler
        self.extensions = extensions
        self.evaluators = evaluators
        self.device = device
        self.epoch = init_epoch
        self.total_epoch = None
        self.history = {}

        #AMP scaler(GPUの場合のみ使用)
        # deviceが'cuda'であることと、configで有効になっているかを確認
        self.use_amp = (self.device.type == 'cuda') and self.cfg.get('amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("[INFO] AMP (自動混合精度) :有効")

    def train(self, epochs, val_loader=None):
        start_time = time.time()
        start_epoch = self.epoch
        self.total_epoch = epochs
        self.history["train"] = []
        self.history["validation"] = []

        print('-----Training Started-----')
        self.train_setup()
        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            # loss is a scalar and self.epoch is incremented in the step function
            # (i.e. self.epoch = epoch + 1)
            #1エポックの学習処理
            loss = self.step()
            self.history["train"].append({'epoch':self.epoch, 'loss':loss})

            # 検証処理
            val_loss = None
            if val_loader is not None:
                vallosses = self.eval(val_loader)
                self.history["validation"].append({'epoch':self.epoch, **vallosses})
                val_loss = vallosses["loss"]
            # スケジューラ更新
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                            self.scheduler.step(val_loss)
                elif not getattr(self.scheduler, 'is_warmup_scheduler', False): # warmup以外
                    self.scheduler.step()
                else:
                    self.scheduler.step()

            self.extend()

        self.train_cleanup()
        print('-----Training Finished-----')

        return self.net

    def step(self):
        self.net.train()
        loss_meter = AverageMeter()
        # 最初に勾配をゼロにする
        self.optimizer.zero_grad()
        # configからaccumulation_stepsを取得。なければ1
        accumulation_steps = self.cfg.get('accumulation_steps', 1) 

        for i, batch in enumerate(self.dataloader):
            # collate_fnによってNoneが返される場合を考慮
            if batch is None: continue
            inputs , targets , _ = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if self.use_amp: #AMP利用
                with autocast(enabled = self.use_amp):
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)
                    if accumulation_steps > 1:
                        # 勾配をステップ数で割って正規化
                        loss = loss / accumulation_steps
                #scalerを利用して勾配をスケーリングし、逆伝播　←なぜ9月27日
                self.scaler.scale(loss).backward()
            else:#通常学習
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / accumulation_steps
                loss.backward()
            
            # 指定したステップ数に一回だけ、重みを更新する
            if (i + 1) % accumulation_steps == 0:
                if self.use_amp:
                    # 勾配をスケーリング解除してからclip
                    self.scaler.unscale_(self.optimizer)
                    nn_utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                     # 勾配適用
                    # optimizerを更新
                    # scalerを使ってoptimizer.step()を呼び出す
                    self.scaler.step(self.optimizer)
                # 次のイテレーションのためにscalerを更新
                    self.scaler.update()
                else:
                    # AMPを使わない場合もclipしてから更新
                    nn_utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    self.optimizer.step()
                # 次のイテレーションのために勾配をリセット
                self.optimizer.zero_grad()

            loss_meter.update(loss.item() * accumulation_steps, number=inputs.size(0))
        #if self.scheduler is not None:
            #self.scheduler.step()
        self.epoch += 1
        ave_loss = loss_meter.average

        return ave_loss

    def eval(self, val_loader=None):
        if self.evaluators is None:
            return

        self.net.eval()
        for evaluator in self.evaluators:
            evaluator.initialize()

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.net(inputs)

                for evaluator in self.evaluators:
                    evaluator.eval_batch(outputs, targets)

        hist_dict = dict()
        for evaluator in self.evaluators:
            result = evaluator.finalize()
            hist_dict = dict(**hist_dict, **result)

        return hist_dict

    def extend(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            if extension.trigger(self):
                extension(self)
        return
    
    def train_setup(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            extension.initialize(self)
        return
    
    def train_cleanup(self) -> typing.NoReturn:
        if self.extensions is None:
            return

        for extension in self.extensions:
            extension.finalize(self)
        return


class Evaluator(ABC):
    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def eval_batch(self) -> typing.NoReturn:
        raise NotImplementedError()
    
    @abstractmethod
    def finalize(self) -> dict:
        raise NotImplementedError()


class LossEvaluator(Evaluator):
    def __init__(self, criterion, criterion_name="loss"):
        super().__init__()
        self.loss_meter = None
        self.criterion = criterion
        self.criterion_name = criterion_name

    def initialize(self):
        self.loss_meter = AverageMeter()

    def eval_batch(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.loss_meter.update(loss.item(), number=outputs.size(0))

    def finalize(self):
        return {"loss": self.loss_meter.average}


class AccuracyEvaluator(Evaluator):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        self.class_correct = None
        self.class_total = None
    
    def initialize(self):
        self.class_correct = list(0. for i in range(len(self.classes)))
        self.class_total = list(0. for i in range(len(self.classes)))

    def eval_batch(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == targets)

        for i in range(len(targets)):
            label = targets[i]
            self.class_correct[label] += c[i].item()
            self.class_total[label] += 1

    def finalize(self):
        class_accuracy = [c / t for c, t in zip(self.class_correct, self.class_total)]
        total_accuracy = sum(self.class_correct) / sum(self.class_total)

        hist_dict = {'total acc': total_accuracy}
        hist_dict = dict(**hist_dict, **{str(self.classes[i]): class_accuracy[i] for i in range(len(self.classes))})
        return hist_dict
