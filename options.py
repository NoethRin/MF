import argparse
from typing import Tuple

def parse_common_args(parser):
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--gpus', type=str, default="1, 3")
    parser.add_argument('--subject', type=int, default=0)
    parser.add_argument('--model_type', type=str, default="meg")
    parser.add_argument('--data_type', type=str, default="cas")
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--task', type=str, default="meg")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_size', type=int, default=128)
    return parser

def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--optim', type=str, default="adam")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--betas', type=Tuple[float, float], default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--pa', type=str, default="emb")
    return parser

def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--mode', type=str, default="test")
    return parser

def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args

def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args