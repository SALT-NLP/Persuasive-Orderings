import argparse

parser = argparse.ArgumentParser(description='Hierachy VAE')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default = 6)
parser.add_argument('--batch-size-u', type=int, default=32)
parser.add_argument('--val-iteration', type=int, default=120)

parser.add_argument('--n-highway-layers', type=int, default=0)
parser.add_argument('--encoder-layers', type=int, default=1)
parser.add_argument('--generator-layers', type=int, default=1)
parser.add_argument('--bidirectional', type=bool, default=False)

parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--encoder-hidden-size', type=int, default=128)
parser.add_argument('--generator-hidden-size', type=int, default=128)
parser.add_argument('--z-size', type=int, default=64)

parser.add_argument('--gpu', default='2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled-data', type=int, default=100,
                    help='Number of labeled data')
parser.add_argument('--n-unlabeled-data', type=int, default=-
                    1, help='Number of unlabeled data')

parser.add_argument('--data-path', type=str,
                    default='./kiva/', help='path to data folders')
parser.add_argument('--max-seq-num', type=int, default=6,
                    help='max sentence num in a message')
parser.add_argument('--max-seq-len', type=int, default=64,
                    help='max sentence length')

parser.add_argument('--word-dropout', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--rec-coef', type=float, default=1)
parser.add_argument('--predict-weight', type=float, default=1)
parser.add_argument('--class-weight', type=float, default=5)
parser.add_argument('--kld-weight-y', type=float, default=1)
parser.add_argument('--kld-weight-z', type=float, default=1)
parser.add_argument('--kld-y-thres', type=float, default=1.4)


parser.add_argument('--warm-up', default='False', type=str)
parser.add_argument('--hard', type=str, default='False')
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--tau-min', type=float, default=0.4)
parser.add_argument('--anneal-rate', type=float, default=0.01)

parser.add_argument('--tsa-type', type=str, default='exp')

z = "--epochs 50 --batch-size 8 --batch-size-u 64 --val-iteration 120 --gpu 1 --max-seq-len 64 --max-seq-num 7 --data-path ./data/borrow/ --rec-coef 1 --predict-weight 0 --class-weight 5 --kld-weight-y 1 --kld-weight-z 1 --word-dropout 0.8 --kld-y-thres 1.5 --warm-up False --tsa-type no --hard True --n-labeled-data 900"

args = parser.parse_args(z.split(" "))