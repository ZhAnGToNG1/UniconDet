import torch
import torch.nn as nn
import time
import datetime

class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batches):
        outputs = self.model(batches[0]['input'])
        loss, scalar_status = self.loss(outputs, batches)
        loss_distill = self.model.guidance_loss(batches[0]['input'])
        scalar_status['distill'] = loss_distill
        loss = loss + loss_distill
        return loss, scalar_status

class UnifyTrainer_Distill(object):
    def __init__(self, cfg, model, loss):
        self.cfg = cfg
        self.model = model
        self.losser = loss
        self.model_with_loss = ModelWithLoss(model, loss)
        if cfg.train.dp_training:
            self.model_with_loss = torch.nn.DataParallel(self.model_with_loss, cfg.train.gpus, None, dim=0).to('cuda')
        else:
            self.model_with_loss = self.model_with_loss.to('cuda')

    def to_cuda(self, batches):
        for i in range(len(batches)):
            batch = batches[i]
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=torch.device('cuda'), non_blocking=True)
        return batches

    def train(self, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.model_with_loss.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1
            batch = self.to_cuda(batch)
            loss, loss_stats = self.model_with_loss(batch)

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                recorder.record('train')








