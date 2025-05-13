from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter
# from torchviz import make_dot
# from torchinfo import summary
# from your_model import YourModel  # 导入你的模型
import argparse
import datetime
import json
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import random
import time
import numpy as np
import ruamel_yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path
import spacy
#allow_tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchvision as tv
import utils
from dataset import create_dataset, create_sampler, create_loader
# from models.model_person_search import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import torch
# from models.model_person_search import ALBEF
# from models.baseline_model_person_search_t import ALBEF
from models.distill_single_model_recode import ALBEF

# 创建一个假输入
# x = torch.randn(1, 3, 224, 224)
# model = YourModel()
# logging.info(f'Loading pretrained {model_name} from OpenAI.')
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/PS_cuhk_pedes.yaml')
parser.add_argument('--output_dir', default='output/cuhk-pedes')
parser.add_argument('--checkpoint', default='')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--eval_mAP', action='store_true', help='whether to evaluate mAP')
parser.add_argument('--text_encoder', default='/workspace/zl/hf-mirror/distilbert-base-uncased')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=True, type=bool)
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

summary(model, input_size=(1, 3, 224, 224))  # Batch size 1, 3 channels, 224x224
# print(model)
