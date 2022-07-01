from tqdm.auto import tqdm
import wandb


import torch
from torch import nn

from typing import Dict

class ModelFit:
    def __init__(self,
                 model: nn.Module,
                 device: str,
                 criterion,
                 optimizator,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 n_epoch: int,
                 classification: bool,
                 wandb_config: Dict = None
                 ):
        """

        :param model:
        :param device:
        :param criterion:
        :param optimizator:
        :param train_loader:
        :param val_loader:
        :param n_epoch:
        :param classification:
        :param wandb_config:
        """

        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizator = optimizator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epoch = n_epoch
        self.classification = classification
        self.wandb_config = wandb_config if wandb_config else {}

    def train(self, epoch: int):
        self.model.train()
        train_epoch_loss = torch.empty(0).to(self.device)
        batch_num = 0
        for batch in tqdm(self.train_loader,
                                desc=f"Train in process..., epoch {epoch}",
                                leave = False):

            data, label = batch['sample'].to(self.device), batch['target'].to(self.device)
            y_pred = self.model(data)
            loss = self.criterion(y_pred.squeeze(1), label)
            loss = torch.sqrt(loss)
            loss.backward()
            self.optimizator.step()
            self.optimizator.zero_grad()

            train_epoch_loss = torch.cat((train_epoch_loss, loss.unsqueeze(0)))

            wandb.log(
                {
                    'rmse per batch on train': loss,
                    'batch': batch_num
                }
            )
            batch_num += 1

        wandb.log(
            {
                'mean loss per epoch on train': train_epoch_loss.mean(),
                'varience loss per epoch on train': torch.var(train_epoch_loss),
                'epoch': epoch
            }
        )

    def evaluate(self, epoch):
        self.model.eval()
        val_epoch_loss = torch.empty(0).to(self.device)



        with torch.no_grad():
            batch_num = 0
            for batch in tqdm(self.val_loader,
                                desc=f"Evaluate in process..., epoch {epoch}",
                                leave = False):
                data, label = batch['sample'].to(self.device), batch['target'].to(self.device)
                y_pred = self.model(data)
                loss = self.criterion(y_pred.squeeze(1), label)

                val_epoch_loss = torch.cat((val_epoch_loss, loss.unsqueeze(0)))
                wandb.log(
                    {
                        'rmse per batch on evaluate': loss,
                        'batch': batch_num
                    }
                )
                batch_num += 1

            wandb.log(
                {
                    'mean loss per epoch on evaluate': val_epoch_loss.mean(),
                    'varience loss per epoch on evaluate': torch.var(val_epoch_loss),
                    'epoch': epoch
                }
            )



    def fit(self, run_name: str):
        wandb.init(
            project="HW1",
            name=run_name,
            config=self.wandb_config,
        )

        wandb.watch(
            self.model,
            criterion=self.criterion,
            log="all",
            log_freq=1000,
            log_graph=(True)
        )

        for epoch in range(self.n_epoch):
            self.train(epoch)
            self.evaluate(epoch)
            print("Training completed...")

    def predict(self, data):
        return self.model(data)

    def save(self, filename):
        torch.save(self.model.state_dict(), f"{filename}.pth")

    def load(self, filename):
        self.model.load_state_dict(torch.load(f"{filename}.pth"))