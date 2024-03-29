from . import *


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)

        return b.mean()

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x,label,class_num):
        # lalel = #
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)

        return b.mean()

class Entropy_each(nn.Module):
    def __init__(self):
        super(Entropy_each, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)

        return b
        
# def make_weights_for_balanced_classes(annotations_path, file_list,nclasses=5):
#     count = np.zeros((nclasses))

#     for filename in file_list:
#         annotations = np.load(annotations_path+filename)
#         count += np.bincount(annotations,minlength=nclasses)

#     weight_per_class = [0.] * nclasses
#     N = float(sum(count))
#     for i in range(nclasses):
#         weight_per_class[i] = N/float(count[i])

#     return weight_per_class , count

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True,weights=None,no_of_classes=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.no_of_classes =no_of_classes
        self.weights = weights
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        if self.weights != None:
            labels_one_hot = F.one_hot(target, self.no_of_classes).float() # one-hot Encoding
            weights = self.weights.unsqueeze(0)# (5) -> (1,5) [1,1,1,1,1]->[[1,1,1,1,]]
            weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot # label에 해당하는 위치의 weight값
            weights = weights.sum(1)
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1,self.no_of_classes) # 정답 class의 weight로 해당 batch의 모든 class weight 일치시킴
            #print(weights)
        else:
            weights = 1

        target = target.view(-1, 1)
        if self.weights != None:
            weights = weights.gather(1, target)

        logpt = F.log_softmax(input)

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = weights * -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()



class MeanFalseError_loss(nn.Module):
    def __init__(self):
        super(MeanFalseError_loss,self).__init__()
    def forward(self, input, target):
        loss = (input - target) ** 2
        return loss.mean(axis=1)

class MeanFalseError(nn.Module):
    def __init__(self,class_num=5):
        super(MeanFalseError,self).__init__()
        self.MeanFalseError_loss = MeanFalseError_loss()

    def forward(self, input, target):
        input = F.softmax(input)
        target = np.eye(self.class_num)[target]
        loss = self.MeanFalseError_loss(input,target)

        return loss.sum()


class MeanFalseError_squared(nn.Module):
    def __init__(self, class_num=5):
        super(MeanFalseError_squared, self).__init__()
        self.MeanFalseError_loss = MeanFalseError_loss()

    def forward(self, input, target):
        input = F.softmax(input)
        target = np.eye(self.class_num)[target]
        loss = self.MeanFalseError_loss(input, target) ** 2

        return loss.sum()

class dice_loss(nn.Module):
    def __init__(self,smooth = 1.):
        super(dice_loss, self).__init__()
        self.smooth = smooth
    def forward(self, input, target):
        num = input.size(0)
        input = F.log_softmax(input)
        m1 = input.view(num,-1).float()
        # pred 값에 대해서 softmax를 해야되는지 파악!
        #m1 = F.softmax(m1)
        m2 = target.view(num,-1).float()
        intersection = (m1 * m2).sum().float()
        loss = (1 - ((2. * intersection + self.smooth)/(m1.sum() + m2.sum() + self.smooth)))**2
        return loss.mean()

class dice_cross_loss(nn.Module):
    def __init__(self,smooth=1.):
        super(dice_cross_loss,self).__init__()
        self.loss_dice = dice_loss(smooth=smooth)

    def forward(self,input,target):
        dl = self.loss_dice(input,target) # dice loss
        cel = F.cross_entropy(input,target) # cross Entropy loss
        return 0.9*cel + 0.1*dl


def CB_focal_loss(labels, logits, alpha, gamma):

    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


# saples_per_cls = 샘플의 비율
# no_of_classes = 클래스의 수
#
class CB_loss(nn.Module):
    def __init__(self,samples_per_cls, no_of_classes, loss_type, beta=0.9999, gamma=2.0):
        super(CB_loss,self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

        no_of_classes = 6
        if no_of_classes == 6:
            self.effective_num = 1.0 - np.power(beta, samples_per_cls[:5])
        else:
            self.effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        print(self.effective_num)
        self.weights = (1.0 - self.beta) / np.array(self.effective_num)

        self.weights = np.append(self.weights,0.)
        print(self.weights)
        self.weights = self.weights / np.sum(self.weights) * self.no_of_classes

        self.weights = torch.tensor(self.weights).float()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.weights = self.weights.to(device)

    def forward(self,logits,labels):
        labels_one_hot = F.one_hot(labels, self.no_of_classes).float() # one-hot Encoding
        weights = self.weights.unsqueeze(0)# (5) -> (1,5) [1,1,1,1,1]->[[1,1,1,1,]]
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot # label에 해당하는 위치의 weight값
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.no_of_classes) # 정답 class의 weight로 해당 batch의 모든 class weight 일치시킴

        if self.loss_type == "focal":
            cb_loss = CB_focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_1_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean") 
     
        return (0.5 * loss)

class JSD_temperal(nn.Module):
    
    def __init__(self,T=3):
        super().__init__()
        self.T = T
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits/self.T, dim=1)
        net_2_probs=  F.softmax(net_2_logits/self.T, dim=1)
        m = (net_1_probs + net_2_probs) / 2
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits/self.T,dim=1), m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(net_2_logits/self.T,dim=1), m, reduction="batchmean") 
        # print(f'loss = {0.5*loss}')
        return (0.5 * loss)

class KL_divergence(nn.Module):
    
    def __init__(self):
        super(KL_divergence, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), net_2_probs, reduction="batchmean") 
     
        return (0.5 * loss)

class KL_divergence_temperal(nn.Module):
    
    def __init__(self,T=3):
        super(KL_divergence_temperal, self).__init__()
        self.T = T
    def forward(self, net_1_logits, net_2_logits):
        # net_1_probs =  F.softmax(net_1_logits/self.T, dim=1)
        net_2_probs=  F.softmax(net_2_logits/self.T, dim=1)

        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits/self.T, dim=1), net_2_probs, reduction="batchmean") 
     
        return (0.5 * loss)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07,epsilon=1e-6):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.epsilon = epsilon
    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # feature shape = [batch, 2, feature_size] 
        # [:,0,:] => data augmentation 1 
        # [:,1,:] => data augmentation 2

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # [batch, 2, feature] => [batch*2, feature]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one': # data augmentation using one 
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all': # data augmentation using more than two
            anchor_feature = contrast_feature # same
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # ([batch * anchor count, feature] matmul [feature, batch * anchor count] / tau)
        # anchor_feature = anchor feature
        # contrastive_feature = positive and negative features
        anchor_dot_contrast = torch.div( 
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # 모든 sample에 대해서 matmul연산

        # for numerical stability
        # print(anchor_dot_contrast)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
        # exit(1)
        # 성능 안정화를 위한 norm 작업
        logits = anchor_dot_contrast - logits_max.detach()
        # print('====mask====',mask.shape)
        # print(mask)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('====mask repeat====',mask.shape)
        # print(mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print('====logit_mask====',logits_mask.shape)
        # print(logits_mask)

        # mask => self remove
        mask = mask * logits_mask
        # print(np.log(exp(1)))
        # exit(1)
        # compute log_prob
        # 분모 value
        exp_logits = torch.exp(logits) * logits_mask  # remove self about matmul at all samples

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # Contrastive Loss

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = ((mask * log_prob).sum(1) + self.epsilon) / (mask.sum(1) + self.epsilon)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss