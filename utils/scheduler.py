from . import *


class LearningRateWarmUP_restart_changeMax(object):
    def __init__(self, optimizer, warmup_iteration, cosine_decay_iter,target_lr=0.1, gamma=0.8,after_scheduler=None,two_param=0):
        self.optimizer = optimizer
        # print(self.optimizer)
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.cur_iteration_decay = 0
        self.cosine_decay_iter = cosine_decay_iter
        self.gamma = gamma
        self.two_param = two_param
    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter)))/float(self.warmup_iteration)
        for index,param_group in enumerate(self.optimizer.param_groups):
            if index == self.two_param:
                param_group['lr'] = warmup_lr
    def step(self, cur_iteration):
        cur_iteration += 1
        # print((cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter))))
        if (cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter))) == 1:
            for index,param_group in enumerate(self.optimizer.param_groups):
                if index == self.two_param:
                    param_group['initial_lr'] = self.target_lr
            # self.optimizer.param_groups[self.two_param]['initial_lr'] = self.target_lr
            self.after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cosine_decay_iter)
        if (cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter))) < self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        elif (cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter))) < self.warmup_iteration + self.cosine_decay_iter:
            # print('cosine : ',cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter)))
            self.after_scheduler.step(cur_iteration-(self.cur_iteration_decay*(self.warmup_iteration+self.cosine_decay_iter))-self.warmup_iteration)
        else:
            # self.after_scheduler.step(cur_iteration - self.warmup_iteration)
            self.cur_iteration_decay += 1
            self.target_lr = self.target_lr * self.gamma
            # self.after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.warmup_iteration)

class LearningRateWarmUP_restart(object):
    def __init__(self, optimizer, warmup_iteration,cosine_decay_iter, target_lr=0.1,after_scheduler=None):
        self.optimizer = optimizer
        # print(self.optimizer)
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.cur_iteration_decay = 0
        self.cosine_decay_iter = cosine_decay_iter
    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr * float(
            cur_iteration - (self.cur_iteration_decay * (self.warmup_iteration + self.cosine_decay_iter))) / float(
            self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr
    def step(self, cur_iteration):
        cur_iteration += 1
        # print(cur_iteration,self.cur_iteration_decay,(cur_iteration-self.cur_iteration_decay) / (self.warmup_iteration*2))
        if (cur_iteration - (self.cur_iteration_decay * (self.warmup_iteration + self.cosine_decay_iter))) < self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        elif (cur_iteration - (self.cur_iteration_decay * (
                self.warmup_iteration + self.cosine_decay_iter))) <= self.warmup_iteration + self.cosine_decay_iter:
            self.after_scheduler.step(cur_iteration - (self.cur_iteration_decay * (
                        self.warmup_iteration + self.cosine_decay_iter)) - self.warmup_iteration)
        else:
            # self.after_scheduler.step(cur_iteration - self.warmup_iteration)
            self.cur_iteration_decay += 1

class LearningRateWarmUP(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        cur_iteration += 1
        if cur_iteration < self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)


def check_warmup():
    v = torch.zeros(10)
    lr = 0.1
    total_iter = 200
    warmup_iter = 20
    cosine_decay_iter = 80

    optim = torch.optim.SGD([v], lr=lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, cosine_decay_iter+1)
    # scheduler = LearningRateWarmUP_restart_changeMax(optimizer=optim,
    #                                warmup_iteration=warmup_iter,
    #                                cosine_decay_iter=cosine_decay_iter,
    #                                gamma=0.8,
    #                                target_lr=lr,
    #                                after_scheduler=scheduler_cosine)

    scheduler = LearningRateWarmUP_restart(optimizer=optim,
                                   warmup_iteration=warmup_iter,
                                   cosine_decay_iter=cosine_decay_iter,
                                   target_lr=lr,
                                   after_scheduler=scheduler_cosine)
    # scheduler = LearningRateWarmUP(optimizer=optim,
    #                                warmup_iteration=warmup_iter,
    #                                target_lr=lr,
    #                                after_scheduler=scheduler_cosine)
    x_iter = [0]
    y_lr = [0.]

    for iter in range(0, total_iter):
        scheduler.step(iter)
        print("iter: ", iter, " ,lr: ", optim.param_groups[0]['lr'])

        optim.zero_grad()
        optim.step()

        x_iter.append(iter)
        y_lr.append(optim.param_groups[0]['lr'])

    plt.plot(x_iter, y_lr, 'b')
    plt.legend(['learning rate'])
    plt.xlabel('iteration')
    plt.ylabel('learning rate')
    plt.savefig('/home/eslab/A.png')




# check_warmup()