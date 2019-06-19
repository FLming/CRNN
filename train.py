import os
import argparse
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import alphabet
import dataset
import crnn
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', default='./data/train_lmdb', help='path to training dataset')
parser.add_argument('--valid_root', default='./data/valid_lmdb', help='path to validation dataset')
parser.add_argument('--seed', type=int, default=0, help='manual random seed')
parser.add_argument('--height', type=int, default=32, help='input height of network')
parser.add_argument('--width', type=int, default=128, help='input width of network')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')

parser.add_argument('--adam', action='store_true')
parser.add_argument('--rmsprop', action='store_true')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, not used by adadealta')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--save_path', type=str, default='models')
args = parser.parse_args()

def prepare_dataloader():
    transform = transforms.Compose([transforms.Resize([args.height, args.width]), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
    # 适用与输入大小不怎么变化的情况，变化巨大的情况可能会降低效率
    torch.backends.cudnn.benchmark = True

    train_set = dataset.lmdbDataset(root=args.train_root, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    valid_set = dataset.lmdbDataset(root=args.valid_root, transform=transform)
    validloader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    return trainloader, validloader

def train(trainloader, crnn, converter, criterion, optimizer):
    running_loss = 0.0
    started = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = crnn(inputs)

        log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
        input_lengths = torch.full(size=(inputs.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)
        target, target_lengths = converter.encode(labels)
        loss = criterion(log_probs, target, input_lengths, target_lengths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            finished = time.time()
            print('[{}, {:5d}] loss: {:.8f} time: {:.2f}s'.format(epoch + 1, i + 1, running_loss / 100, finished - started))
            running_loss = 0.0
            started = finished

def validate(validloader, crnn, converter):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = crnn(inputs)

            _, predicted = outputs.max(2)

            predicted = predicted.transpose(1, 0).contiguous().view(-1)
            input_lengths = torch.full(size=(inputs.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            sim_preds = converter.decode(predicted.data, input_lengths.data, raw=False)
            targets = [i.decode('utf-8', 'strict') for i in labels]
            for sim_pred, target in zip(sim_preds, targets):
                total += 1
                if sim_pred == target:
                    correct += 1

    print('[{}] Accuracy of the network on the {} validation images: {:.2%}'.format(epoch + 1, total, correct / total))
    with open('{}/train.log'.format(args.save_path), 'a') as f:
        f.write('[{}] Accuracy of the network on the {} validation images: {:.2%}\n'.format(epoch + 1, total, correct / total))


if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    num_classes = len(alphabet.alphabet) + 1
    converter = utils.StrLabelConverter(alphabet.alphabet)

    trainloader, validloader = prepare_dataloader()

    crnn = crnn.CRNN(num_classes).to(device)

    criterion = torch.nn.CTCLoss().to(device)
    if args.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=args.lr)
    elif args.rmsprop:
        optimizer = optim.RMSprop(crnn.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adadelta(crnn.parameters())

    if args.pretrained != '':
        print('loading pretrained model from {}'.format(args.pretrained))
        crnn.load_state_dict(torch.load(args.pretrained))

    crnn.train()
    for epoch in range(args.num_epoch):
        
        train(trainloader, crnn, converter, criterion, optimizer)

        if epoch % args.eval_epoch == 0:
            print('-------------------- eval --------------------')
            crnn.eval()
            validate(validloader, crnn, converter)
            crnn.train()
        if epoch % args.save_epoch == 0:
            torch.save(crnn.state_dict(), '{}/crnn_{}.pth'.format(args.save_path, epoch + 1))
