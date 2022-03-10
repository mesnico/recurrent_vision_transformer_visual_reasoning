import os
import argparse
import time
import glob
import pickle
import subprocess
import shlex
import io
import pprint
import importlib
import json

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup

Image.warnings.simplefilter('ignore')

np.random.seed(4)
torch.manual_seed(4)

torch.backends.cudnn.benchmark = True
NORMALIZE_TRANSFORM = torchvision.transforms.Normalize(mean=[0.9884, 0.9884, 0.9884],
                                                       std=[0.1068, 0.1068, 0.1068])
RESIZE_TRANSFORM = torchvision.transforms.Resize(224)

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', default='./',
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--data_path_validation', default=None,
                    help='optional path to dataset folder containing the desired validation set')
parser.add_argument('-o', '--output_path', default='runs/test',
                    help='path for storing ')
parser.add_argument('--model', default='cornet_z',
                    help='which model to train')
parser.add_argument('--times', default=5, type=int,
                    help='number of time steps to run the model (only R and S models)')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size')
parser.add_argument('--subdivisions', default=1, type=int,
                    help='number of subdivisions of the minibatch')
parser.add_argument('--lr', '--learning_rate', default=.1, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=10, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('--normalize', action='store_true', default=False, help="Performs image normalization")
parser.add_argument('--resize', action='store_true', default=False, help="Resize input to 224")
parser.add_argument('--clip-grad', type=int, default=-1, help="Clip the gradient to a maximum norm. -1 to disable")
parser.add_argument('--training_imgs', type=int, default=-1, help="Number of training images. -1 to use all")
parser.add_argument('--depth', type=int, default=6, help="Number of layers for the ViT")
parser.add_argument('--u-depth', type=int, default=6, help="Number of layers for the Universal Transformer")
parser.add_argument('--data-augmentation', action='store_true', default=False, help="Performs data augmentation")
parser.add_argument('--pretrain', type=str, choices=['rn', 'resnet-2-8', 'resnet-2-16', 'resnet-3'], help='CNN Pretrain Mode')
parser.add_argument('--warmup', type=int, default=0, help="Warmup (in iterations)")

FLAGS, FIRE_FLAGS = parser.parse_known_args()

if FLAGS.resize:  #'resnet' in FLAGS.model or 'alexnet' in FLAGS.model or 'densenet' in FLAGS.model: # or 'vgg' in FLAGS.model:
    print('RESIZED!')

if FLAGS.normalize:
    print('NORMALIZED!')


def set_gpus(n=1):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    gpus = subprocess.run(shlex.split(
        'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True, stdout=subprocess.PIPE).stdout
    gpus = pandas.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
    # gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        visible = [int(i)
                   for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpus = gpus[gpus['index'].isin(visible)]
    gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        [str(i) for i in gpus['index'].iloc[:n]])


#if FLAGS.ngpus > 0:
#    set_gpus(FLAGS.ngpus)


def get_model(pretrained=False, model_name=FLAGS.model):
    map_location = None if FLAGS.ngpus > 0 else 'cpu'
    if 'transformer' not in model_name:
        model_module = importlib.import_module('cornet')
        model = getattr(model_module, model_name.lower())
        model = model(pretrained=pretrained, map_location=map_location)
    else:
        model_module = importlib.import_module('transformer')
        model = getattr(model_module, model_name.lower())
        model = model(pretrained=pretrained, map_location=map_location, depth=FLAGS.depth, u_depth=FLAGS.u_depth)

    if FLAGS.ngpus > 0:
        model = model.cuda()
    return model


def train(restore_path=None,  # useful when you want to restart training
          save_train_epochs=.1,  # how often save output during training
          save_val_epochs=.5,  # how often save output during validation
          save_model_epochs=5,  # how often save model weigths
          save_model_secs=60*10  # how often save model (in sec)
          ):

    model = get_model(pretrained=FLAGS.pretrain)
    tb_logger = SummaryWriter(log_dir=FLAGS.output_path, comment='')

    # nn.utils.clip_grad_norm(model.parameters(), 0)
    trainer = SVRTTrain(model)
    validator = SVRTVal(model)

    start_epoch = 0
    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        start_epoch = ckpt_data['epoch']
        model.load_state_dict(ckpt_data['state_dict'])
        trainer.optimizer.load_state_dict(ckpt_data['optimizer'])
        print('Checkpoint restored!')

    records = []
    recent_time = time.time()

    nsteps = len(trainer.data_loader)
    if save_train_epochs is not None:
        save_train_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_train_epochs) * nsteps).astype(int)
    if save_val_epochs is not None:
        save_val_steps = (np.arange(0, FLAGS.epochs + 1,
                                    save_val_epochs) * nsteps).astype(int)
    if save_model_epochs is not None:
        save_model_steps = (np.arange(0, FLAGS.epochs + 1,
                                      save_model_epochs) * nsteps).astype(int)

    results = {'meta': {'step_in_epoch': 0,
                        'epoch': start_epoch,
                        'wall_time': time.time()}
               }
    best_val_acc = 0
    for epoch in tqdm.trange(0, FLAGS.epochs + 1, initial=start_epoch, desc='epoch'):
        data_load_start = np.nan
        for step, data in enumerate(tqdm.tqdm(trainer.data_loader, desc=trainer.name)):
            data_load_time = time.time() - data_load_start
            global_step = epoch * len(trainer.data_loader) + step

            if save_val_steps is not None:
                if global_step in save_val_steps:
                    results[validator.name] = validator()
                    for k, v in results[validator.name].items():
                        tb_logger.add_scalar(validator.name + '/' + k, v, global_step)
                    trainer.model.train()

            if FLAGS.output_path is not None:
                # records.append(results)
                # if len(results) > 1:
                #     pickle.dump(records, open(FLAGS.output_path + 'results.pkl', 'wb'))

                ckpt_data = {}
                ckpt_data['flags'] = FLAGS.__dict__.copy()
                ckpt_data['epoch'] = epoch
                ckpt_data['state_dict'] = model.state_dict()
                ckpt_data['optimizer'] = trainer.optimizer.state_dict()

                # if save_model_secs is not None:
                #     if time.time() - recent_time > save_model_secs:
                #         torch.save(ckpt_data, FLAGS.output_path +
                #                    'latest_checkpoint.pth.tar')
                #         recent_time = time.time()

                if save_model_steps is not None:
                    if global_step in save_model_steps:
                        # save latest_checkpoint
                        torch.save(ckpt_data, os.path.join(FLAGS.output_path, 'latest_checkpoint.pth.tar'))

                        if results[validator.name]['top1'] > best_val_acc:
                            # save best model
                            torch.save(ckpt_data, os.path.join(FLAGS.output_path, 'best_checkpoint.pth.tar'))
                            best_val_acc = results[validator.name]['top1']

            else:
                if len(results) > 1:
                    pprint.pprint(results)

            if epoch < FLAGS.epochs:
                frac_epoch = (global_step + 1) / len(trainer.data_loader)
                record = trainer(frac_epoch, *data)
                record['data_load_dur'] = data_load_time
                results['meta'] = {'step_in_epoch': step + 1,
                                    'epoch': frac_epoch,
                                    'wall_time': time.time()}
                for k, v in results['meta'].items():
                    tb_logger.add_scalar('meta/'+k, v, global_step)
                if save_train_steps is not None:
                    if step in save_train_steps:
                        results[trainer.name] = record
                        for k, v in results[trainer.name].items():
                            tb_logger.add_scalar(trainer.name + '/' + k, v, global_step)



            data_load_start = time.time()


# def test(layer='decoder', sublayer='avgpool', imsize=224):
#     """
#     Suitable for small image sets. If you have thousands of images or it is
#     taking too long to extract features, consider using
#     `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.
#
#     Kwargs:
#         - layers (choose from: V1, V2, V4, IT, decoder)
#         - sublayer (e.g., output, conv1, avgpool)
#     """
#     model = get_model(pretrained=True)
#     transform = torchvision.transforms.Compose([
#                     torchvision.transforms.Resize(imsize),
#                     torchvision.transforms.ToTensor(),
#                     normalize,
#                 ])
#     model.eval()
#
#     def _store_feats(layer, inp, output):
#         """An ugly but effective way of accessing intermediate model features
#         """
#         _model_feats.append(np.reshape(output, (len(output), -1)).numpy())
#
#     model_layer = getattr(getattr(model._modules['module'], layer), sublayer)
#     model_layer.register_forward_hook(_store_feats)
#
#     model_feats = []
#     with torch.no_grad():
#         model_feats = []
#         fnames = sorted(glob.glob(os.path.join(FLAGS.data_path, '*.*')))
#         if len(fnames) == 0:
#             raise f'No files found in {FLAGS.data_path}'
#         for fname in tqdm.tqdm(fnames):
#             try:
#                 im = Image.open(fname).convert('RGB')
#             except:
#                 raise f'Unable to load {fname}'
#             im = transform(im)
#             im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
#             _model_feats = []
#             model(im)
#             model_feats.append(_model_feats[0])
#         model_feats = np.concatenate(model_feats)
#
#     if FLAGS.output_path is not None:
#         fname = f'CORnet-{FLAGS.model}_{layer}_{sublayer}_feats.npy'
#         np.save(os.path.join(FLAGS.output_path, fname), model_feats)




def calculate_mean_var():
    mean = 0.
    std = 0.
    nb_samples = 0.

    dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'val'),
            torchvision.transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=FLAGS.batch_size,
                                              shuffle=False,
                                              num_workers=FLAGS.workers,
                                              pin_memory=True)

    for data in tqdm.tqdm(data_loader):
        data = data[0] # get the image leaving the target apart
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print('Dataset mean: {}; Dataset variance: {}'.format(mean, std))


def validate(restore_path=None):
    validate_different_angles(restore_path, num_angles=0)


def validate_different_angles(restore_path=None, num_angles=0):
    model = get_model()
    results = []
    if num_angles==0:
        angles = [0]
    else:
        angles = np.linspace(0, 360, num_angles, endpoint=False)
    # nn.utils.clip_grad_norm(model.parameters(), 0)

    if restore_path is not None:
        ckpt_data = torch.load(restore_path)
        print('Loaded epoch {}'.format(ckpt_data['epoch']))
        model.load_state_dict(ckpt_data['state_dict'])

    for angle in tqdm.tqdm(angles, desc='angles'):
        validator = SVRTVal(model, angle)
        data = validator()
        data['angle'] = angle
        results.append(data)

        # if FLAGS.output_path is not None:
        #     if len(results) > 1:
        #         pickle.dump(results, open(FLAGS.output_path + 'results_varying_angles.pkl', 'wb'))
        # else:
        pprint.pprint(results)


def test_convergence(experiment_json_file):
    # test the convergence of all the models described in the input json file
    results = []

    print('Experiment JSON file: {}'.format(experiment_json_file))

    with open(experiment_json_file, 'r') as f:
        experiments = json.load(f)

    print('Running checkpoints sanity check...')
    for exp in tqdm.tqdm(experiments):
        try:
            model = get_model(model_name=exp['network'])
            ckpt_data = torch.load(exp['checkpoint'])
            # print('Loaded epoch {}'.format(ckpt_data['epoch']))
            model.load_state_dict(ckpt_data['state_dict'])
        except Exception as e:
            print('Error loading model for experiment {}'.format(exp["output_full"]))

    # cycle over all the experiments list and run the model on the test set
    for exp in tqdm.tqdm(experiments):
        print('Evaluating model {}'.format(exp['output_full']))
        try:
            model = get_model(model_name=exp['network'])
            ckpt_data = torch.load(exp['checkpoint'])
            # print('Loaded epoch {}'.format(ckpt_data['epoch']))
            model.load_state_dict(ckpt_data['state_dict'])

            dataset_path = os.path.join(FLAGS.data_path_validation, 'results_problem_{}'.format(exp['problem']))
            validator = SVRTVal(model, image_set='test', normalize=exp['normalize'], resize=exp['resize'], dataset_path=dataset_path)
            data = validator()
        except Exception as e:
            print(e)
            continue

        out_exp = exp.copy()
        data = {'test_'+k: v for k, v in data.items()}
        out_exp.update(data)
        results.append(out_exp)

        # dump on the output json after every experiment
        out_file = experiment_json_file.replace('.json', '') + '_out.json'
        with open(out_file, 'w') as f:
            json.dump(results, f)

    print('DONE!')


def test_generalization(experiment_json_file):
    # test the convergence of all the models described in the input json file
    results = []

    print('Experiment JSON file: {}'.format(experiment_json_file))

    with open(experiment_json_file, 'r') as f:
        experiments = json.load(f)

    # first of all, sanity check for all the checkpoints. If an error is thrown, there is something wrong
    print('Running checkpoints sanity check...')
    for exp in tqdm.tqdm(experiments):
        model = get_model(model_name=exp['network'])
        ckpt_data = torch.load(exp['checkpoint'])
        # print('Loaded epoch {}'.format(ckpt_data['epoch']))
        model.load_state_dict(ckpt_data['state_dict'])

    # cycle over all the experiments list and run the model on the test set
    for exp in tqdm.tqdm(experiments):
        print('Testing model {}'.format(exp['output_full']))
        problem = exp['problem']
        if problem == 1:
            test_problem_list = [5, 20, 21]
        elif problem == 21:
            test_problem_list = [1, 5, 20]
        else:
            print('Warning!! Problem {} not expected!'.format(problem))
            continue

        try:
            model = get_model(model_name=exp['network'])
            ckpt_data = torch.load(exp['checkpoint'])
            # print('Loaded epoch {}'.format(ckpt_data['epoch']))
            model.load_state_dict(ckpt_data['state_dict'])
        except Exception as e:
            print(e)
            continue

        for test_problem in tqdm.tqdm(test_problem_list):
            print('Testing problem {}'.format(test_problem))

            dataset_path = os.path.join(FLAGS.data_path_validation, 'results_problem_{}'.format(test_problem))
            validator = SVRTVal(model, image_set='test', normalize=exp['normalize'], resize=exp['resize'], dataset_path=dataset_path)
            data = validator()

            out_exp = exp.copy()
            data = {'test_'+k: v for k, v in data.items()}
            out_exp.update(data)
            out_exp.update({'test_problem': test_problem})
            results.append(out_exp)

            # dump on the output json after every experiment
            out_file = experiment_json_file.replace('.json', '') + '_out.json'
            with open(out_file, 'w') as f:
                json.dump(results, f)

    print('DONE!')


def test_generalization_single_experiment(resume_path, training_problem):
    print('Testing model {}'.format(FLAGS.model))
    problem = training_problem
    if problem == 1:
        test_problem_list = [1, 5, 20, 21]
    elif problem == 21:
        test_problem_list = [1, 5, 20, 21]
    else:
        print('Warning!! Problem {} not expected!'.format(problem))
    _test(resume_path, test_problem_list)


def test_convergence(resume_path, training_problem):
    print('Testing model {}'.format(FLAGS.model))
    problem = training_problem
    test_problem_list = [problem]
    _test(resume_path, test_problem_list)


def _test(resume_path, test_problem_list):
    try:
        model = get_model(pretrained=FLAGS.pretrain, model_name=FLAGS.model)
        ckpt_data = torch.load(resume_path)
        # print('Loaded epoch {}'.format(ckpt_data['epoch']))
        model.load_state_dict(ckpt_data['state_dict'])
    except Exception as e:
        print(e)

    for test_problem in tqdm.tqdm(test_problem_list):
        print('Testing problem {}'.format(test_problem))

        dataset_path = os.path.join(FLAGS.data_path_validation, 'results_problem_{}'.format(test_problem))
        validator = SVRTVal(model, image_set='test', dataset_path=dataset_path)
        data = validator()

        # out_exp = exp.copy()
        out_exp = {'test_' + k: v for k, v in data.items()}
        print('Model: {}; Problem {}: {}'.format(resume_path, test_problem, out_exp))

    print('DONE!')

class SVRTTrain(object):

    def __init__(self, model):
        self.name = 'train'
        self.model = model
        self.data_loader = self.data()
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  FLAGS.lr,
        #                                  momentum=FLAGS.momentum,
        #                                  weight_decay=FLAGS.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        self.optimizer.zero_grad()
        self.lr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=FLAGS.step_size)
        self.warmup = pytorch_warmup.LinearWarmup(self.optimizer, warmup_period=FLAGS.warmup) if FLAGS.warmup > 0 else None
        self.loss = nn.CrossEntropyLoss()
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()
        self.current_subdiv = 0

    def data(self):
        # transforms = []
        # if FLAGS.resize:
        #     transforms.append(RESIZE_TRANSFORM)
        # transforms.append(torchvision.transforms.ToTensor())
        # if FLAGS.normalize:
        #     transforms.append(NORMALIZE_TRANSFORM)
        # load images

        transform_list = []

        if FLAGS.resize:
            transform_list.append(torchvision.transforms.Resize(224))

        if FLAGS.data_augmentation:
            # data augmentation
            transform_list.extend([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_list.append(torchvision.transforms.ToTensor())

        transform = torchvision.transforms.Compose(transform_list)

        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.data_path, 'train'),
            transform)
        training_images = FLAGS.training_imgs
        if training_images > 0:
            neg_samples = list(range(0, training_images // 2))
            pos_samples = [n + (len(dataset) // 2) for n in range(0, training_images // 2)]
            indexes = neg_samples + pos_samples
            dataset = torch.utils.data.Subset(dataset, indexes)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size // FLAGS.subdivisions,
                                                  shuffle=True,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)
        return data_loader

    def __call__(self, frac_epoch, inp, target):
        start = time.time()

        self.lr.step(epoch=frac_epoch)
        if self.warmup is not None:
            self.warmup.dampen()
        if FLAGS.ngpus > 0:
            target = target.cuda(non_blocking=True)
            inp = inp.cuda()

        record = {}

        if 'transformer' in FLAGS.model:
            loss, output, n_updates = self.model(inp, target)
            record['n_updates'] = n_updates
        else:
            output = self.model(inp)
            loss = self.loss(output, target)

        record['loss'] = loss.item()

        loss /= FLAGS.subdivisions

        record['top1'] = accuracy(output, target, topk=(1,))[0]
        record['top1'] /= len(output)
        # record['top5'] /= len(output)
        record['learning_rate'] = self.lr.get_lr()[0]

        loss.backward()
        if FLAGS.clip_grad > 0:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), FLAGS.clip_grad)

        self.current_subdiv += 1

        if self.current_subdiv == FLAGS.subdivisions:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_subdiv = 0

        record['dur'] = time.time() - start
        return record


class SVRTVal(object):

    def __init__(self, model, angle=None, resize=FLAGS.resize, normalize=FLAGS.normalize, image_set='val', dataset_path=FLAGS.data_path_validation):
        self.resize = resize
        self.normalize = normalize
        self.angle = angle
        self.image_set = image_set
        self.dataset_path = dataset_path
        self.name = image_set
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        if FLAGS.ngpus > 0:
            self.loss = self.loss.cuda()

    def data(self):
        # transforms = []
        # '''if self.angle is not None:
        #     transforms.append(torchvision.transforms.Pad(80, fill=(255, 255, 255)))
        #     transforms.append(torchvision.transforms.Lambda(
        #         lambda img: torchvision.transforms.functional.rotate(img, self.angle, resample=Image.BILINEAR)
        #     ))
        #     transforms.append(torchvision.transforms.CenterCrop(192))'''
        # if self.resize:
        #     transforms.append(RESIZE_TRANSFORM)
        # transforms.append(torchvision.transforms.ToTensor())
        # if self.normalize:
        #     transforms.append(NORMALIZE_TRANSFORM)

        transform_list = []

        if FLAGS.resize:
            transform_list.append(torchvision.transforms.Resize(224))

        if FLAGS.data_augmentation:
            transform_list.extend([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            transform_list.append(torchvision.transforms.ToTensor())

        transform = torchvision.transforms.Compose(transform_list)

        if self.dataset_path is not None:
            val_dataset = self.dataset_path
            print('Using validation from {}'.format(val_dataset))
        else:
            val_dataset = FLAGS.data_path

        dataset = torchvision.datasets.ImageFolder(
            os.path.join(val_dataset, self.image_set),
            transform)
        val_images = 18000
        neg_samples = list(range(0, val_images // 2))
        pos_samples = [n + (len(dataset) // 2) for n in range(0, val_images // 2)]
        indexes = neg_samples + pos_samples
        dataset = torch.utils.data.Subset(dataset, indexes)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'n_updates': 0, 'loss': 0, 'top1': 0, 'top5': 0}
        with torch.no_grad():
            for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
                if FLAGS.ngpus > 0:
                    target = target.cuda(non_blocking=True)
                    inp = inp.cuda()
                if 'transformer' in FLAGS.model:
                    _, output, n_updates = self.model(inp, target)
                    record['n_updates'] += (n_updates * FLAGS.batch_size)
                else:
                    output = self.model(inp)
                '''img = inp.numpy()
                img = img[0]
                img = np.moveaxis(img, 0, -1)
                plt.imshow(img)
                plt.savefig('test.jpg')'''

                record['loss'] += self.loss(output, target).item()
                p1 = accuracy(output, target, topk=(1,))[0]
                record['top1'] += p1
                # record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)
