# coding: utf-8
import argparse
import time
import math
import os
import logging
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

# experimental dense embedding
parser.add_argument('--dense-embedding', action='store_true',
                    help='use a dense embedding layer')
parser.add_argument('--dense-embedding-type', type=str, default='default',
                    help='initial value of the dense embedding layer')
parser.add_argument('--dense-embedding-n', type=int, default=400,
                    help='size of the dense embedding layer')
parser.add_argument('--layer-adapt-lowerbound', type=float, default=1.0,
                    help='the lower bound of the layer adaptive rescaling')
parser.add_argument('--layer-adapt-upperbound', type=float, default=200.0,
                    help='the upper bound of the layer adaptive rescaling')
parser.add_argument('--stage2', type=int, default=41,
                    help='the 1st epoch of the 2nd stage')
parser.add_argument('--lr2', type=float, default=5,
                    help='initial learning rate of the 2nd stage')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument("--log", type=str, help="dir of the log file", default='wikitext2_lm')


args = parser.parse_args()

print(args)

# logging
filehandler = logging.FileHandler(args.log + '_' + args.save + '.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, args.dense_embedding, args.dense_embedding_n, args.dense_embedding_type).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(stage = 1):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for name, p in model.named_parameters():
            if p.requires_grad:
                # # debug
                # if batch % args.log_interval == 0 and batch > 0:
                #     gnorm, wnorm = torch.norm(p.grad).item(), torch.norm(p.data).item()
                #     print('name=%25s, ||g||=%1.6f, ||w||=%3.6f, ||g||/||w||=%1.6f' % (name, gnorm, wnorm, gnorm / wnorm))
                rescale = 1.0
                if layer_adapt_lowerbound != 1.0 or layer_adapt_upperbound != 1.0:
                    rescale = torch.norm(p.data) / torch.norm(p.grad)
                    rescale = torch.maximum(rescale, torch.tensor(layer_adapt_lowerbound))
                    rescale = torch.minimum(rescale, torch.tensor(layer_adapt_upperbound))
                    # rescale = torch.norm(p.data) / torch.norm(p.grad)
                p.data.add_(p.grad, alpha=-lr * rescale)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #         'loss {:5.2f} | ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // args.bptt, lr,
            #     elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | layer_ub {:03.0f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr, layer_adapt_upperbound, 
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

layer_adapt_lowerbound = args.layer_adapt_lowerbound
layer_adapt_upperbound = args.layer_adapt_upperbound
lr_decay_counter = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        if epoch == args.stage2: lr = args.lr2
        epoch_start_time = time.time()
        train(stage = 2 if epoch >= args.stage2 else 1)
        val_loss = evaluate(val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                    val_loss, math.exp(val_loss)))
        # print('-' * 89)
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logger.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if lr_decay_counter < 1 or layer_adapt_upperbound == 1.0:
                lr /= 4.0
            else:
                layer_adapt_upperbound /= 4.0
                layer_adapt_upperbound = max(layer_adapt_upperbound, 1.0)
            lr_decay_counter += 1
            # lr *= 0.1
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
