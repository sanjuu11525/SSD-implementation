import torch.optim as optim
from torch.autograd import Variable
from multibox_loss import MultiboxLoss
from data_coder import DataCoder

class Trainer():
    def __init__(self, parameter, model):
        self.n_epoch = parameter['n_epoch']
        self.device = parameter['device']
        self.model = model.to(self.device)
        self.milestones = parameter['milestones']
        learning_rate  = parameter['learning_rate']
        weight_decay   = parameter['weight_decay']
        momentum = parameter['momentum']
        gamma = parameter['gamma']
        
        self.optimizer = optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
        self.criteria = MultiboxLoss()
        if self.milestones is not None:
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, gamma)
        self.DataCoder = DataCoder()
        
    def train(self, loss_total, loss_conf_out, loss_loc_out, trainer_loader):

        num_iter = 0      
        for epoch in range(self.n_epoch + 1):
	    print("epoch:{}".format(epoch))
            if self.milestones is not None:
                self.scheduler.step()
                
            for it, (x_batch, ground_truth) in enumerate(trainer_loader):
                target_box, target_label = self.DataCoder.targetCoder(ground_truth)
                x_batch = Variable(x_batch.to(self.device))
                target_box = target_box.to(self.device)
                target_label = target_label.to(self.device)                
                pred_loc, pred_clf = self.model(x_batch)
                loss_conf, loss_loc = self.criteria(pred_loc, pred_clf, target_box, target_label)
                loss = loss_conf + loss_loc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                num_iter = num_iter + 1
                
                if it % 20 == 0:
                    loss_total.append(loss.data.cpu().numpy())
                    loss_conf_out.append(loss_conf.data.cpu().numpy())
                    loss_loc_out.append(loss_loc.data.cpu().numpy())
                if it % 400 == 0:
                    print('Iteration:{}, loss_total:{}, loss_conf:{}, loss_loc:{}'.format(num_iter, loss.cpu().data.numpy(), loss_conf.cpu().data.numpy(), loss_loc.cpu().data.numpy()))
