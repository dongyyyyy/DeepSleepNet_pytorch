from models.cnn.DeepSleepNet_cnn import *
from utils.dataset.Sleep_edf.edf_to_numpy import *
import torch

from torchsummary import summary

        


if __name__ == '__main__':
    remove_unnessersary_wake()
    # make_dataset()
    # check_wellmade()
    # summary(model.cuda(),(1,3000))


