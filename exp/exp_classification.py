from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa

    def _build_model(self):
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        if self.args.use_graph:
            test_data,test_loader,test_graph_data,test_graph_loader=self._get_data(flag="TEST")
        else:
            test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))
        # self.args.num_class = 3
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        if self.args.use_graph:
            data_set,data_loader,graph_data,graph_data_loader=data_provider(self.args, flag)
            return data_set,data_loader,graph_data,graph_data_loader
        else:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def _calculate_combined_loss(self, outputs, hp, hc, hg, label, criterion):
        # 1. Cross-Entropy Loss
        loss_ce = criterion(outputs, label.long())

        # 2. Pool embeddings to get a single vector per modality
        hp_mean = hp.mean(dim=1) if hp is not None and hp.dim() > 2 else hp
        hc_mean = hc.mean(dim=1) if hc is not None and hc.dim() > 2 else hc
        hg_mean = hg.mean(dim=1) if hg is not None and hg.dim() > 2 else hg
        # hg_mean = hg.squeeze(1) if hg is not None and hg.dim() > 2 else hg
        # print(f"hp_mean: {hp_mean.shape}, hc_mean: {hc_mean.shape}, hg_mean: {hg_mean.shape}")
        
        # 3. Calculate hf (fused representation) for the loss calculation
        available_views = [v for v in [hp_mean, hc_mean, hg_mean] if v is not None]
        
        if not available_views:
            return loss_ce, 0.0, 0.0

        hf_for_loss = torch.mean(torch.stack(available_views), dim=0)

        # 4. Fused Alignment Loss (L_fa)
        loss_fa = 0
        for v_mean in available_views:
            loss_fa += F.mse_loss(hf_for_loss, v_mean)
        
        # 5. Inter-view Agreement Loss (L_ia)
        loss_ia = 0
        if len(available_views) > 1:
            for i in range(len(available_views)):
                for j in range(i + 1, len(available_views)):
                    loss_ia += F.mse_loss(available_views[i], available_views[j])
        
        # 6. Combine losses
        total_loss = loss_ce + self.args.lambda1 * loss_fa + self.args.lambda2 * loss_ia
        return total_loss, loss_fa.item(), loss_ia.item()

    def vali(self, vali_data, vali_loader, criterion,vali_graph_data=None,vali_graph_loader=None):
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, ((batch_x, label, padding_mask),(graph_batch)) in enumerate(zip(vali_loader,vali_graph_loader)):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                graph_batch=graph_batch.to(self.device)
                # print(graph_batch.device)
                if self.swa:
                    if self.args.use_graph:
                        outputs, hp, hc, hg = self.swa_model(batch_x, padding_mask,None,None,graph_batch=graph_batch)
                    else:
                        outputs, hp, hc, hg = self.swa_model(batch_x, padding_mask,None,None,graph_batch=None)
                else:
                    if self.args.use_graph:
                        outputs, hp, hc, hg = self.model(batch_x, padding_mask,None,None,graph_batch=graph_batch)
                    else:
                        outputs, hp, hc, hg = self.model(batch_x, padding_mask,None,None,graph_batch=None)

                # pred = outputs.detach().cpu()
                # loss = criterion(pred, label.long().cpu())
                loss, _, _ = self._calculate_combined_loss(outputs, hp, hc, hg, label, criterion)
                total_loss.append(loss.item())
                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues_onehot, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues_onehot, probs, average="macro"),
        }

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        if self.args.use_graph:
            train_data,train_loader,train_graph_data,train_graph_loader=self._get_data(flag="TRAIN")
            vali_data,vali_loader,vali_graph_data,vali_graph_loader=self._get_data(flag="VAL")
            test_data,test_loader,test_graph_data,test_graph_loader=self._get_data(flag="TEST")
        else:
            train_data, train_loader = self._get_data(flag="TRAIN")
            vali_data, vali_loader = self._get_data(flag="VAL")
            test_data, test_loader = self._get_data(flag="TEST")

        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)
        l=[]
        path = (
            "./checkpoints/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # total_params = sum(p.numel() for p in self.model.parameters())
        # trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # print(f"Total parameters: {total_params}")
        # print(f"Trainable parameters: {trainable_params}")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, ((batch_x, label, padding_mask),(graph_batch)) in enumerate(zip(train_loader,train_graph_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                graph_batch=graph_batch.to(self.device)
                if self.args.use_graph:
                    outputs, hp, hc, hg = self.model(batch_x, padding_mask,None,None,graph_batch=graph_batch)
                else:
                    outputs, hp, hc, hg = self.model(batch_x, padding_mask,None,None,graph_batch=None)
                loss, loss_fa, loss_ia = self._calculate_combined_loss(outputs, hp, hc, hg, label, criterion)
                # loss = criterion(outputs, label.long())
                train_loss.append(loss.item())
                # train_loss_list.append(loss.item())
                # train_fa_list.append(loss_fa)
                # train_ia_list.append(loss_ia)

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if self.args.use_graph:
                vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion,vali_graph_data=vali_graph_data,vali_graph_loader=vali_graph_loader)
                test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion,vali_graph_data=test_graph_data,vali_graph_loader=test_graph_loader)
            else:
                vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
                test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)
            l.append(test_metrics_dict['Accuracy'])
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path + "checkpoint.pth"
        print(max(l))
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if self.args.use_graph:
          vali_data,vali_loader,vali_graph_data,vali_graph_loader=self._get_data(flag='VAL')
          test_data,test_loader,test_graph_data,test_graph_loader=self._get_data(flag='TEST')
        else:
          vali_data, vali_loader = self._get_data(flag="VAL")
          test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            path = (
                "./checkpoints/"
                + self.args.task_name
                + "/"
                + self.args.model_id
                + "/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        if self.args.use_graph:
          vali_loss,val_metrics_dict=self.vali(vali_data,vali_loader,criterion,vali_graph_data,vali_graph_loader)
          test_loss,test_metrics_dict=self.vali(test_data,test_loader,criterion,test_graph_data,test_graph_loader)
        else:
          vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
          test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(setting + "  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return
