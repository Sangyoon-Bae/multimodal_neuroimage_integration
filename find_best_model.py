import torch
import os
import re

# read file
file_dir = '/home/ubuntu/Stella/MLVU_multimodality/Graphormer/exps/abcd-struct/abcd-struct-100-1500/1/lightning_logs/checkpoints'
file_name = 'last.ckpt'
data = os.path.join(file_dir, file_name)

# load checkpoint
checkpoint = torch.load(data)
print(checkpoint['callbacks'])

# it will return score (validation loss) of best model, and you should run test dataset with that model