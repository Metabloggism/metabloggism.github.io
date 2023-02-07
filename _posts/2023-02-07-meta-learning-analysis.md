As in the previous post, we will work in a Notebook from now on. You can find the Notebook file [here](https://colab.research.google.com/drive/1JXOIYy2cKEf5K602_hfrMKc5NsMh8Wvq?usp=sharing).

# Meta-Learning: MAML evaluation and discussion by Metabloggism

In this Notebook we will evaluate the performance of the MAML approach we implemented in our last post. To do so, we need first to correct something we did wrong and we noticed during the development of this Notebook. We will add an explanation about that in an introdury manner. Later, the intention is not to present the results of a paper trying to achieve an impressive performance, but to try to understand how is Meta-Learning contributing in our scenario.

To make that analysis we will use metrics, plots, and different visualization techniques that, not only will allow us to take our target conclusions but will also serve as a reference for Meta-Learning analysis procedures.

## Correcting last post

Let's do a quick rewind of the previous posts before continuing on.

* [First](https://metabloggism.github.io/2022/11/21/meta-learning.html) we reviewed the definition and motivations of Meta-Learning, as well as some State of the Art approaches.

* [Second](https://metabloggism.github.io/2022/12/20/meta-learning-implementation.html), we built some experiments to test Meta-Learning as well as a dummy MAML approach that we trained in the scenario.

If you remember well, in the last post we proved two ways to train a model (with Pytorch and manually) and used both to train at 2 levels (Pytorch itself allows one single training, storing a single gradient per parameter). We ended training a Meta-Learning pipeline with them both.

However, the approach didn't seem to train at all. There is a reason behind that, and we will get hands on now to explain it.

### Why was Meta-Learning not working?



Making it simple, values of the parameters in the network were replaced manually which didn't allow proper gradient tracking. Check the code cells below, corresponding to the cell in the previous post (I just included the Meta-Training cell and the *make_step* and *update_model* methods, the rest of variables and methods are assumed or we will end with another kilometric post, anyways that is just a reminder and all them are included in the previous post).



```
def make_step(model, outputs, labels, update_lr, in_weights):
    loss = criterion(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters())
    out_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grads, in_weights)))
    accuracy = (((1 - outputs) < outputs).float() == labels).sum() / outputs.shape[0]
    return out_weights, loss, accuracy
```





```
def update_model(model, new_weights, param_keys):
    for param, param_key in zip(new_weights, param_keys):
        model._modules[param_key[0]]._parameters[param_key[1]] = param
```





```
model = SimpleNet()
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01
meta_lr = 0.001
n_epochs = 15
n_metaepochs = 2

metaoptimizer = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.9)
param_keys = [(mod, kname) for mod in model._modules for kname in model._modules[mod]._parameters]

for metaepoch in range(n_metaepochs):

    print('===============================')
    print(f'//           Meta-Epoch {metaepoch + 1}       //')    
    print('===============================')

    for mi, metabatch in enumerate(metatrain_loader, 0):  #  Meta-step
        print(f'{mi} updates at Meta-Level')

        running_loss = 0.0  #  At each meta-step, the loss is reset

        initial_weights = model.parameters()

        for pi, problem_loaders in enumerate(metabatch, 0):  #  Problem in the meta-batch

            print(f'- Problem {pi + 1} -')

            problem_loader = problem_loaders['train']
            problem_loader_val = problem_loaders['val']
            ref_label = None

            new_weights = initial_weights

            for epoch in range(n_epochs):  #  Epoch in the problem training

                print(f'Epoch {epoch + 1}')

                val_loss = 0.0
                val_accuracy = 0.0

                for i, data in enumerate(problem_loader, 0):  #  Step in the problem

                    inputs_raw, labels_raw = data
                    inputs = preprocess_inputs(inputs_raw)
                    outputs = model(inputs)
                    if ref_label is None:
                        ref_label = labels_raw[0]   #  On a new problem (1st step) adjust label mapping
                    labels = process_labels(labels_raw, ref_label)

                    new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)
                    update_model(model, new_weights, param_keys)  #  At each step in the problem manually update the model

                    print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation

                    inputs_rawv, labels_rawv = datav
                    inputsv = preprocess_inputs(inputs_rawv)
                    outputsv = model(inputsv)
                    labelsv = process_labels(labels_rawv, ref_label)

                    lossv = criterion(outputsv, labelsv[0])  #  Loss in a validation batch
                    val_loss += lossv.item()
                    val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation

            running_loss += lossv  #  After all epochs (all training process) in a single problem the validation loss is added

            update_model(model, initial_weights, param_keys)  # After the whole train + validation of a problem and the final loss is added, return the model to its original stage in the meta-step 
        
        metastep_loss = running_loss / metabatch_size  #  The added validation losses of all problems in the metabatch are averaged

        metaoptimizer.zero_grad()  #  We perform gradient descent at the Meta-Level over the averaged validation loss
        metastep_loss.backward()
        metaoptimizer.step()

        if (mi + 1) % 1000 == 0:  #  Meta-validation performed every 1000 meta-steps

            print('META-VALIDATION STEP:')

            for mbvi, metabatch_val in enumerate(metaval_loader):  #  Meta-validation meta-step

                if (mbvi + 1) % 10 == 0:

                    print(f'Validation step {mbvi + 1}')
                    
                initial_weights = model.parameters()

                for problem_loaders in metabatch_val:  #  Problem in the meta-validation meta-batch

                    problem_loader = problem_loaders['train']
                    problem_loader_val = problem_loaders['val']
                    ref_label = None
                    new_weights = initial_weights

                    for epoch in range(n_epochs):  #  Epoch in the problem training

                        val_loss = 0.0
                        val_accuracy = 0.0

                        for i, data in enumerate(problem_loader, 0):  #  Step in the problem
                            
                            inputs_raw, labels_raw = data
                            inputs = preprocess_inputs(inputs_raw)
                            outputs = model(inputs)
                            if ref_label is None:
                                ref_label = labels_raw[0]
                            labels = process_labels(labels_raw, ref_label)

                            new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)
                            update_model(model, new_weights, param_keys)  #  Note that we still need to update although being in (Meta-)validation. That is because we are in meta-validation but at the Learning level we are in training stage

                        #    print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                        for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation, as in Meta-Train

                            inputs_rawv, labels_rawv = datav
                            inputsv = preprocess_inputs(inputs_rawv)
                            outputsv = model(inputsv)
                            labelsv = process_labels(labels_rawv, ref_label)
                            
                            lossv = criterion(outputsv, labelsv[0])
                            val_loss += lossv.item()
                            val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                    
                    if (mbvi + 1) % 10 == 0:

                        print(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)

                    update_model(model, initial_weights, param_keys)

            print('END OF META-VALIDATION STEP')
```



Note in the method *update_model* that after the value of the new weights (which actually corresponds to both weights and biases) is computed,  they are explicitly assigned to the model internal attributes. Later on (in the following iteration), these parameters (with explicitly modified values) are used for prediction and corresponding gradient computation, and that is the point where the process fails, as Pytorch's autograd doesn't understand the gradient flow and fails at updating this. Thus, the model is not working at Learning and therefore the Meta-Learning does not progress. The difference with the manual GD algorithm above is that .

So, with that said, we just want an alternative way to assemble the manually computed weights to get the corresponding loss at Learning level that Meta-Learning demands, but allowing also to recover the state of the model at the beginning of each Meta-batch.

### The *Learner* object

Looking back again at the previous post, we first talked about some implementations (both official and unofficial). We mentioned [an unofficial implementation](https://github.com/dragen1860/MAML-Pytorch) that somehow showed the way to Meta-train the model with this Pytorch + manual parallel training (Chelsea Finn's team [official implementation](https://github.com/cbfinn/maml) also did this). That said, we may compare with them the way to proceed for this model update.

Exploring their repo, it seems that the script that runs an experiment like ours is in [this file](https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py). Pulling the string from there, we may notice that they define a [MAML model](https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py) that works with Meta-batches at the Meta-Learningg level. This model needs also to work at the Learning level in the way aforementioned, and it seems that in this code they do so by using a [*Learner* object](https://github.com/dragen1860/MAML-Pytorch/blob/master/learner.py). 

Which is the key point of all that? Well, the part that we are not able to do yet is the Learning part, so we may want to look at what the Learner object does. Looking carefully, we may notice the key difference, which is the fact that the Learner object is able to predict by using specified weights at forward time different than the ones stored at the model. It does so by using Conv2d in [Torch's nn.functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d) instead of plain [nn](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). Model parameters are directly stored in an internal list and layers are directly built at each forward's pass with the shape specified at Learner object's configuration.

Ok, knowing all that we will directly try to use the Learner object at the Learning Level.

### Train with Learner

So, the first step will be to repeat the Learning level GD that we did both with Pytorch''s optimization module and manually, but now with the Learner object (also manual updates).

Before going to the code, we need to point out that as we are using a different Conv2d module, we need to adapt something which is that we will not scale the input images between 0 and 255 but between 0 and 1. This is an issue that I found while training the Learner model and I had to spend several hours until figuring that out!

First let's copy the Learner class and define the Network configuration (as we had it before). Well, first of all actually let's just run the relevant code from the previous Notebook in order to run the interesting parts now (just run and ignore this first cell, it's just repeating old work, focus on the rest).


```python
from google.colab import drive
drive.mount('/content/drive')   
# WRITE PATH WHERE YOU WERE
%cd drive/MyDrive/collab_space/metabloggism/meta-learning

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, BatchSampler
import torchvision
import matplotlib.pyplot as plt

omniglot_raw = torchvision.datasets.Omniglot(root="./dataset/omniglot", download=True, transform=torchvision.transforms.ToTensor())


alphabets = omniglot_raw._alphabets
characters = omniglot_raw._characters


num_alphabets = len(alphabets)
num_characters = len(characters)

class MetaSplit:
  def __init__(self, ratio, total_num_characters):
    self.alphabets = []
    self.num_characters = 0
    self.min_num_characters = total_num_characters * ratio
    self.num_problems = None

metasplits = {'metatrain': MetaSplit(0.7, num_characters),
              'metaval': MetaSplit(0.15, num_characters),
              'metatest': MetaSplit(0.15, num_characters)}

chars_per_alphabet = {alph: [char.split('/')[0] for char in characters].count(alph) for alph in alphabets}

random.shuffle(alphabets)

current_metasplit = 'metatrain'
switch_metasplit_from = {'metatrain': 'metaval', 'metaval': 'metatest'}

for alphabet in alphabets:
  if not metasplits[current_metasplit].num_characters < metasplits[current_metasplit].min_num_characters:
    current_metasplit = switch_metasplit_from[current_metasplit]
  metasplits[current_metasplit].alphabets.append(alphabet)
  metasplits[current_metasplit].num_characters += chars_per_alphabet[alphabet]

for metasplit in metasplits:
  metasplits[metasplit].num_problems = 1/2 * sum([chars_per_alphabet[alph]**2 - chars_per_alphabet[alph] for alph in metasplits[metasplit].alphabets])

metabatch_size = 8
num_metabatches = int(metasplits['metatrain'].num_problems / metabatch_size)

class MetaLoader():
    """
    """
    def __init__(self, base_dataset, metabatch_size, batch_sizes, 
                 chars_per_alphabet, problem_ratios):
        self.base_dataset = base_dataset
        self.metabatch_size = metabatch_size
        self.batch_sizes = batch_sizes
        self.chars_per_alph = chars_per_alphabet
        self.problem_ratios = [0.75, 0.15, 0.1]
        self.problems_per_alph = {}
        self.num_problems = 0
        self.__load_quantitative_info__()
        self.metasampler = BatchSampler(RandomSampler(range(self.num_problems)), 
                                        batch_size=self.metabatch_size, 
                                        drop_last=True)
    
    def __load_quantitative_info__(self):
        for alphb in self.chars_per_alph:
            self.problems_per_alph[alphb] = int((self.chars_per_alph[alphb]**2 - 
                                                self.chars_per_alph[alphb]) / 2)
            self.num_problems += self.problems_per_alph[alphb]
    
    def __has_reached__(self, idx, ctr, current):
        return ctr + current > idx
    
    def __problem_idx_to_samples_idx__(self, problem_idx, alphb, 
                                       prbs_on_prev_alphabets, 
                                       chars_on_prev_alphabets):
        pb_idx_in_alph = problem_idx - prbs_on_prev_alphabets
        ichars_in_alphabet = (int(pb_idx_in_alph / self.chars_per_alph[alphb]), 
                                pb_idx_in_alph % self.chars_per_alph[alphb])
        ichars = tuple([ich + chars_on_prev_alphabets \
                        for ich in ichars_in_alphabet])
        return [sample_idx for charidx in ichars 
                for sample_idx in range(charidx * 20, (charidx + 1) * 20)]
    
    def __build_problem_loader_from_samples__(self, samples_idx):

        random.shuffle(samples_idx)

        train_val_frontier = int(len(samples_idx) * self.problem_ratios[0])
        val_test_frontier = int(train_val_frontier + 
                                len(samples_idx) * self.problem_ratios[1])
        
        samples_idx_train = samples_idx[:train_val_frontier]
        samples_idx_val = samples_idx[train_val_frontier:val_test_frontier]
        samples_idx_test = samples_idx[val_test_frontier:]

        train_sampler = BatchSampler(SubsetRandomSampler(samples_idx_train), 
                                     batch_size=self.batch_sizes['train'], 
                                     drop_last=True)
        val_sampler = BatchSampler(SubsetRandomSampler(samples_idx_val), 
                                   batch_size=self.batch_sizes['val'], 
                                   drop_last=True)
        test_sampler = BatchSampler(SubsetRandomSampler(samples_idx_test), 
                                    batch_size=self.batch_sizes['test'], 
                                    drop_last=True)
        loaders = {'train': DataLoader(dataset=self.base_dataset, 
                                       batch_sampler=train_sampler),
                   'val': DataLoader(dataset=self.base_dataset, 
                                       batch_sampler=val_sampler),
                   'test': DataLoader(dataset=self.base_dataset, 
                                       batch_sampler=test_sampler)}
        return loaders

        
    def __get_problem_loader__(self, problem_idx):
        pbs_ctr = 0
        chars_ctr = 0
        for alphb in self.chars_per_alph:
            if not self.__has_reached__(problem_idx, pbs_ctr, 
                                        self.problems_per_alph[alphb]):
                pbs_ctr += self.problems_per_alph[alphb]
                chars_ctr += self.chars_per_alph[alphb]
            else:
                problem_samples_idx = self.__problem_idx_to_samples_idx__(
                    problem_idx, alphb, pbs_ctr, chars_ctr)
                return self.__build_problem_loader_from_samples__(
                    problem_samples_idx)

    def  __iter__(self):
        for imetabatch, metabatch in enumerate(self.metasampler):
            problem_loaders = []
            for problem_idx in metabatch:
                problem_loaders.append(self.__get_problem_loader__(problem_idx))
            yield problem_loaders

chars_per_alphabet = {split: {alph: [char.split('/')[0] for char in characters].count(alph) for alph in metasplits[split].alphabets} for split in metasplits}

metatrain_loader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=metabatch_size, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metatrain'], problem_ratios = [0.75, 0.15, 0.1])
metaval_loader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=metabatch_size, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metaval'], problem_ratios = [0.75, 0.15, 0.1])
metatest_loader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=1, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metatest'], problem_ratios = [0.75, 0.15, 0.1])

n_epochs = 15

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.conv3 = nn.Conv2d(10, 12, 5)
        self.conv4 = nn.Conv2d(12, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 2 * 2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = x.squeeze()
        return x


def process_labels(labels_raw, ref_label):
  return (labels_raw == ref_label).float()

def preprocess_inputs(inputs):
    return (1- inputs) * 255

def make_step(model, outputs, labels, update_lr, in_weights):
    loss = criterion(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters())
    out_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grads, in_weights)))
    accuracy = (((1 - outputs) < outputs).float() == labels).sum() / outputs.shape[0]
    return out_weights, loss, accuracy

def update_model(model, new_weights, param_keys):
    for param, param_key in zip(new_weights, param_keys):
        model._modules[param_key[0]]._parameters[param_key[1]] = param

toy_metabatch = next(iter(metatrain_loader))
toy_problem_loader = toy_metabatch[0]['train']
toy_problem_loader_val = toy_metabatch[0]['val']
toy_problem_loader_test = toy_metabatch[0]['test']
```

    Mounted at /content/drive
    /content/drive/MyDrive/collab_space/metabloggism/meta-learning
    Files already downloaded and verified



```python
class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
```

    ...



```python
net_config = [
        ('conv2d', [6, 1, 5, 5, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [10, 6, 5, 5, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [12, 10, 5, 5, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [16, 12, 5, 5, 1, 0]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [10, 64]),
        ('relu', [True]),
        ('linear', [1, 10]),
        ('sigmoid', []),
        ('reshape', [])
    ]
```

And do the training procedure


```python
model = Learner(net_config, imgc=1, imgsz=28)
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01

ref_label = None
new_weights = model.parameters()
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}')
    running_loss = 0.0
    val_loss = 0.0
    val_accuracy = 0.0
    for i, data in enumerate(toy_problem_loader, 0):
        inputs_raw, labels_raw = data
        inputs = 1 - inputs_raw
        outputs = model(inputs, new_weights)
        if ref_label is None:
            ref_label = labels_raw[0]
        labels = process_labels(labels_raw, ref_label)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, new_weights)
        new_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grads, new_weights)))
        running_loss += loss.item()
        accuracy = (((1 - outputs) < outputs).float() == labels).sum() / outputs.shape[0]
        print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
    for iv, datav in enumerate(toy_problem_loader_val):
        inputs_rawv, labels_rawv = datav
        inputsv = 1 - inputs_rawv
        outputsv = model(inputsv, new_weights)
        labelsv = process_labels(labels_rawv, ref_label)
        lossv = criterion(outputsv, labelsv)
        val_loss += lossv.item()
        val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()
    print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')

print('Finished Training')
```

    Epoch 1
    Epoch 1, step     1], Loss: 0.7337116003036499, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6828164458274841, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6651676893234253, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.7054024835427603, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.7743954658508301, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6243588924407959, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6621766090393066, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6930027008056641, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.5869386196136475, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.7064928412437439, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.7682352066040039, Accuracy: 0.75
    Epoch 3, VALIDATION], Loss: 0.681134819984436, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6635424494743347, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6362461447715759, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6927491426467896, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6729293664296468, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6286272406578064, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6477007269859314, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.691921055316925, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6657606065273285, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6737616062164307, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.7064082622528076, Accuracy: 0.75
    Epoch 6, step     3], Loss: 0.6098929643630981, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6592579881350199, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.7201383113861084, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6647733449935913, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.5731081366539001, Accuracy: 0.25
    Epoch 7, VALIDATION], Loss: 0.653910239537557, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6345923542976379, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.607459545135498, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6215931177139282, Accuracy: 0.75
    Epoch 8, VALIDATION], Loss: 0.6490383495887121, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.5579589605331421, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6945024132728577, Accuracy: 0.875
    Epoch 9, step     3], Loss: 0.6135469675064087, Accuracy: 0.875
    Epoch 9, VALIDATION], Loss: 0.6399873395760854, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.5632861852645874, Accuracy: 0.875
    Epoch 10, step     2], Loss: 0.6671784520149231, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.627591073513031, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6310261487960815, Accuracy: 0.8333333134651184
    Epoch 11
    Epoch 11, step     1], Loss: 0.6116976141929626, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6371555328369141, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6205811500549316, Accuracy: 1.0
    Epoch 11, VALIDATION], Loss: 0.6219043781359991, Accuracy: 0.8333333134651184
    Epoch 12
    Epoch 12, step     1], Loss: 0.564954936504364, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6245361566543579, Accuracy: 1.0
    Epoch 12, step     3], Loss: 0.6186693906784058, Accuracy: 0.875
    Epoch 12, VALIDATION], Loss: 0.6143342554569244, Accuracy: 0.8333333134651184
    Epoch 13
    Epoch 13, step     1], Loss: 0.5977659225463867, Accuracy: 0.875
    Epoch 13, step     2], Loss: 0.6437636017799377, Accuracy: 0.875
    Epoch 13, step     3], Loss: 0.509320855140686, Accuracy: 0.875
    Epoch 13, VALIDATION], Loss: 0.6065680583318075, Accuracy: 0.8333333134651184
    Epoch 14
    Epoch 14, step     1], Loss: 0.5873907208442688, Accuracy: 0.875
    Epoch 14, step     2], Loss: 0.6376876831054688, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.5768213272094727, Accuracy: 1.0
    Epoch 14, VALIDATION], Loss: 0.5984243750572205, Accuracy: 0.8333333134651184
    Epoch 15
    Epoch 15, step     1], Loss: 0.6466023921966553, Accuracy: 0.75
    Epoch 15, step     2], Loss: 0.6546764969825745, Accuracy: 1.0
    Epoch 15, step     3], Loss: 0.4909670352935791, Accuracy: 0.875
    Epoch 15, VALIDATION], Loss: 0.5915585309267044, Accuracy: 0.8333333134651184
    Finished Training


### Meta-Learning with *Learner* object

In the following cell we will finally train again the Meta-Learning pipeline using the method we have been developing in the previous sections.


```python
printlines = []

model = Learner(net_config, imgc=1, imgsz=28)
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01
meta_lr = 0.0001
n_epochs = 15
n_metaepochs = 2

metaoptimizer = optim.Adam(model.parameters(), lr=meta_lr)

for metaepoch in range(n_metaepochs):

    printlines.append('===============================')
    printlines.append(f'//           Meta-Epoch {metaepoch + 1}       //')    
    printlines.append('===============================')
    print('===============================')
    print(f'//           Meta-Epoch {metaepoch + 1}       //')    
    print('===============================')

    for mi, metabatch in enumerate(metatrain_loader, 0):  #  Meta-step
        print(mi)
        printlines.append(f'{mi} updates at Meta-Level')
        print(f'{mi} updates at Meta-Level')

        running_loss = 0.0  #  At each meta-step, the loss is reset

        # No need to store initial weights

        for pi, problem_loaders in enumerate(metabatch, 0):  #  Problem in the meta-batch

            printlines.append(f'- Problem {pi + 1} -')
            print(f'- Problem {pi + 1} -')

            problem_loader = problem_loaders['train']
            problem_loader_val = problem_loaders['val']
            ref_label = None

            new_weights = model.parameters()

            for epoch in range(n_epochs):  #  Epoch in the problem training

                printlines.append(f'Epoch {epoch + 1}')
                print(f'Epoch {epoch + 1}')

                val_loss = 0.0
                val_accuracy = 0.0

                for i, data in enumerate(problem_loader, 0):  #  Step in the problem

                    inputs_raw, labels_raw = data
                    inputs = 1 - inputs_raw
                    outputs = model(inputs, new_weights)
                    if ref_label is None:
                        ref_label = labels_raw[0]   #  On a new problem (1st step) adjust label mapping
                    labels = process_labels(labels_raw, ref_label)

                    new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                    #  As the prediction is intrinsically done with the new weights, no need to actually update the model at the Learning Level

                    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
                    print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation

                    inputs_rawv, labels_rawv = datav
                    inputsv = 1 - inputs_rawv
                    outputsv = model(inputsv, new_weights)
                    labelsv = process_labels(labels_rawv, ref_label)

                    lossv = criterion(outputsv, labelsv)  #  Loss in a validation batch
                    val_loss += lossv.item()
                    val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                printlines.append(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation
                print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation

            running_loss += lossv  #  After all epochs (all training process) in a single problem the validation loss is added

            # Again, no need to update the model to the initial weights 
        
        metastep_loss = running_loss / metabatch_size  #  The added validation losses of all problems in the metabatch are averaged

        metaoptimizer.zero_grad()  #  We perform gradient descent at the Meta-Level over the averaged validation loss
        metastep_loss.backward()
        metaoptimizer.step()

        if (mi + 1) % 1000 == 0:  #  Meta-validation performed every 1000 meta-steps

            printlines.append('META-VALIDATION STEP:')
            print('META-VALIDATION STEP:')

            for mbvi, metabatch_val in enumerate(metaval_loader):  #  Meta-validation meta-step

                if (mbvi + 1) % 10 == 0:

                    printlines.append(f'Validation step {mbvi + 1}')
                    print(f'Validation step {mbvi + 1}')

                for problem_loaders in metabatch_val:  #  Problem in the meta-validation meta-batch

                    problem_loader = problem_loaders['train']
                    problem_loader_val = problem_loaders['val']
                    ref_label = None
                    new_weights = model.parameters()

                    for epoch in range(n_epochs):  #  Epoch in the problem training

                        val_loss = 0.0
                        val_accuracy = 0.0

                        for i, data in enumerate(problem_loader, 0):  #  Step in the problem
                            
                            inputs_raw, labels_raw = data
                            inputs = 1 - inputs_raw
                            outputs = model(inputs)
                            if ref_label is None:
                                ref_label = labels_raw[0]
                            labels = process_labels(labels_raw, ref_label)

                            new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                        #    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                        for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation, as in Meta-Train

                            inputs_rawv, labels_rawv = datav
                            inputsv = 1 - inputs_rawv
                            outputsv = model(inputsv)
                            labelsv = process_labels(labels_rawv, ref_label)
                            
                            lossv = criterion(outputsv, labelsv)
                            val_loss += lossv.item()
                            val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                    
                    if (mbvi + 1) % 10 == 0:

                        printlines.append(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)
                        print(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)

            printlines.append('END OF META-VALIDATION STEP')
            print('END OF META-VALIDATION STEP')





```

    ...
	
    Epoch 15, step     2], Loss: 0.5346571803092957, Accuracy: 1.0
    Epoch 15, step     3], Loss: 0.4873807728290558, Accuracy: 1.0
    Epoch 15, VALIDATION], Loss: 0.6720514297485352, Accuracy: 0.8333333134651184
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6898062825202942, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.7031591534614563, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6912919282913208, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6870008011658987, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.7087008357048035, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6882234811782837, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6735674738883972, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6823924879233042, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.70091313123703, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.680241048336029, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6792828440666199, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6770347853501638, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6692278385162354, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6950821280479431, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6796243786811829, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6731089850266775, Accuracy: 0.8333333134651184
    Epoch 5

We finally stopped the process after 10 hours of Meta-training (and including one validation step).



## Meta-Learning performance analysis

### Parsing the training

The results of both regular training and Meta-training are stored in files. In the first case I copied it manually while in the second case note that the code stores the lines to include, so we just need to do this:


```python
with open('metalearning_001.txt', 'w') as f:
    for line in printlines:
        f.write(f"{line}\n")
```

Now let's include file parsing to store in an structured way the information in both cases


```python
meta_learning_training_output_lines = open('metalearning_001.txt', 'r').readlines()
```


```python
learning_training = []

crt_epoch = 0
crt_step = 0
new_epoch = False

for line in learning_training_output_lines:
    if not line.startswith('Epoch '):
        break
    elif all([chr.isdigit() for chr in line[6:-1]]):
        crt_epoch = int(line[6:-1])
        new_epoch = True
    else:
        line_parts = line.split(', ')
        line_epoch_str = line_parts[0]
        line_step_str = line_parts[1]
        line_loss_str = line_parts[2]
        line_accuracy_str = line_parts[3]
        epoch = int(line_epoch_str.split(' ')[1])
        assert epoch == crt_epoch
        assert line_loss_str.split(': ')[0] == 'Loss'
        assert line_accuracy_str.split(': ')[0] == 'Accuracy'
        loss_value = float(line_loss_str.split(': ')[1])
        accuracy_value = float(line_accuracy_str.split(': ')[1])
        if line_step_str.strip(']') == 'VALIDATION':
            learning_training[-1]['validation'] = {'loss': loss_value, 
                                                   'accuracy': accuracy_value}
        else:
            step = int(line_step_str.strip(']').split(' ')[-1])
            assert (step == (crt_step + 1)) or (new_epoch and step == 1)
            crt_step = step
            learning_training.append({'epoch': epoch, 'step': step, 
                                      'train': {'loss': loss_value, 
                                                'accuracy': accuracy_value}})
```


```python
metalearning_training = []
crt_metaepoch = 0
crt_metastep = 0
crt_problem = 0
new_metaepoch = False
new_metastep = False
on_metaval = False

for line in meta_learning_training_output_lines:
    print(line)
    if 'Meta-Epoch' in line:
        metaepoch_line_core = line.strip('//')
        metaepoch = int([word for word in metaepoch_line_core.split(' ') if word][1])
        assert metaepoch == (crt_metaepoch + 1)
        crt_metaepoch = metaepoch
        new_metaepoch = True
    elif ' updates at Meta-Level' in line and line.split(' ')[0].isdigit():
        metastep = int(line.split(' ')[0]) + 1
        assert (metastep == (crt_metastep + 1)) or (new_metaepoch and metastep == 1)
        crt_metastep = metastep
        metalearning_training.append({'Meta-Epoch': crt_metaepoch, 
                                      'Meta-step': crt_metastep,
                                      'Problems' : []})
        new_metastep = True
        new_metaepoch = False
    elif 'Problem' in line and line.split(' ')[1] == 'Problem' and line.split(' ')[2].isdigit():
        problem_num = int(line.split(' ')[2])
        assert (problem_num == (crt_problem + 1)) or (new_metastep and problem_num == 1)
        metalearning_training[-1]['Problems'].append([])
        crt_problem = problem_num
        crt_epoch = 0
        crt_step = 0
        new_epoch = False
    elif line.startswith('Epoch '):
        if all([chr.isdigit() for chr in line[6:-1]]):
            crt_epoch = int(line[6:-1])
            new_epoch = True
        else:            
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            epoch = int(line_epoch_str.split(' ')[1])
            assert epoch == crt_epoch
            assert line_loss_str.split(': ')[0] == 'Loss'
            assert line_accuracy_str.split(': ')[0] == 'Accuracy'
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            if line_step_str.strip(']') == 'VALIDATION':
                metalearning_training[-1]['Problems'][-1][-1]['validation'] = \
                    {'loss': loss_value, 'accuracy': accuracy_value}
            else:
                step = int(line_step_str.strip(']').split(' ')[-1])
                assert (step == (crt_step + 1)) or (new_epoch and step == 1)
                crt_step = step
                metalearning_training[-1]['Problems'][-1].append(
                    {'epoch': epoch, 'step': step, 'train': {'loss': loss_value, 
                                                   'accuracy': accuracy_value}})
            new_epoch = False
    elif 'META-VALIDATION STEP:' in line:
        metalearning_training[-1]['Meta-Validation'] = []
        on_metaval = True
    elif line.startswith('Validation step'):
        assert on_metaval
        valstep = int(line.split(' ')[-1])
        metalearning_training[-1]['Meta-Validation'].append(
            {'step': valstep, 'problems': []})
    elif line.startswith('Last epoch, VALIDATION'):
        if on_metaval:
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            metalearning_training[-1]['Meta-Validation'][-1]['problems'].append(
                {'loss': loss_value, 'accuracy': accuracy_value})
        else:
            continue
    elif 'END OF META-VALIDATION STEP' in line:
        on_metaval = False
    else:
        continue



```

    ...
    



```python
with open('metalearning_001.txt', 'w') as f:
    for line in printlines:
        f.write(f"{line}\n")
```


```python
meta_learning_training_output_lines = open('metalearning_001.txt', 'r').readlines()
```

### Plots

#### Initial Train Loss Learning evolution

In this plot we will just verify how did the Train Loss evolve in a regular Learning problem before Meta-Learning.


```python
plt.plot([pstep['train']['loss'] for pstep in metalearning_training[0]['Problems'][0]])
```




    [<matplotlib.lines.Line2D at 0x7f08050be490>]




    
![Imgur](https://i.imgur.com/VUKBEwn.png)
    


If you have some basis in ML, you will know that in this example Learning is probably not evolving at its best. One should probably lower the LR. However that may depend on the problem. If you remind in the first post we already talked about some Meta-Learning approaches that will learn the best optimizer settings for this type of problems. As the model will be (meta-)evolving, the best configuration at the first meta-epoch may not be the same as in the last meta-epoch. Given that, we will skp this part and assume we are using a proper initialization. Moreover, what we will be doing in general is actually Meta-Learning the best initialization to solve these problems with this configuration. If inidividual trainings make no sense and are not able to learn anything we will re-do this configuration but it is not the idea.

#### Train Loss evolutions by Meta-Step (at each problem)

First, we will remove the last (possibly incomplete) meta-step to avoid conflicts and define a method to smooth the Loss functions.


```python
metalearning_training = metalearning_training[:-1]
```


```python
def smooth_plot(pline, wsize):
    smoothed_pline = []
    for ipl in range(1, len(pline) + 1):
        smoothed_pline.append(sum(pline[max(0, ipl - wsize):ipl]) / min(wsize, ipl))
    return smoothed_pline
```

So now we will plot a grid of Loss functions. Each position contains the Loss of a given problem. In vertical there is the Meta-Training evolution while in horizontal there are the problems of the meta-batch.


```python
plt.figure(figsize=(25,50))
for iims in range(13):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(13, 8, (iims * 8) + iprb + 1)
        plt.plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]])
```


    
![Imgur](https://i.imgur.com/YW2d3FR.png)
    


Note here that in some cases the Loss progresses smoothly while in others there is a bit of unstability. That prooves the aforementioned thought that a single configuration may work differently on different problems (from a given meta-step) given a common initialization. It may be like saying that each problem is in a different point of the training. However we may repeat it smoothed to verify that the general tendency of the losses is descending regularly.


```python
plt.figure(figsize=(25,50))
for iims in range(13):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(13, 8, (iims * 8) + iprb + 1)
        plt.plot(smooth_plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]], 5))
```


    
![Imgur](https://i.imgur.com/0ZOKxw8.png)
    


So it seems fine.

#### Train Loss Meta-Evolution

In this plot we will verify how the Train Loss avolves by Meta-epoch. Note that it is not our target Loss (ours is the Validation Loss instead) so it will not necessarily descend (it may actually increase).


```python
plt.plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f07ef8195e0>]




    
![Imgur](https://i.imgur.com/bUdtqKi.png)
    



```python
plt.plot(smooth_plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f07ef7dedc0>]




    
![Imgur](https://i.imgur.com/wlviyto.png)
    


#### Validation Loss evolutions by Meta-Step

At this point we will focus in our target (Meta-Training) Loss, i.e. validation Loss. We will plot it as we did with the Train Loss, but in this case not all individual train steps contain a validation run so we will just plot the validations done. Also for that reason we will not include smoothed plots, since plots are short (and hopefully statistically meaningful) enough. Tendency should also be to lower, but there may be, especially at the first Meta-steps, problems with wrong behaviours. We expect that these phenomena will reduce while the Meta-training is advancing.


```python
plt.figure(figsize=(25,200))
for ims, metastep in enumerate(metalearning_training):
    if ims % 10 != 0:
        continue
    for iprb in range(8):
        plt.subplot(130, 8, (int(ims / 10) * 8) + iprb + 1)
        plt.plot([prbstep['validation']['loss'] for prbstep in metastep['Problems'][iprb] if 'validation' in prbstep.keys()])

```


    
![Imgur](https://i.imgur.com/e43bVJ6.png)
    


#### Validation Loss evolution $\equiv$ Meta-Train Loss

Nothing to add, just the Meta-Train Loss evolution. Ideally should behave as a normal Train Loss.


```python
plt.plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f07dbc013a0>]




    
![Imgur](https://i.imgur.com/DvwFKx2.png)
    


And the smoothed version


```python
plt.plot(smooth_plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f07db8aedc0>]




    
![Imgur](https://i.imgur.com/jFanzzG.png)
    


As you can see, the Meta-Train Loss is very unstable, and the general tendency tends to raise. There is [several information](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/) on this pehenomenon when using a high static LR.

### Experiments

#### Lower Meta-LR

Following a common procedure in the Meta-Learning level as we would use in the Learning level, we will use a lower (in this case, Meta-)LR. This way we will try to capture a better tendency in general (less sensible to step variations). So we will repeat the same visualizations with our new experiment.


```python
printlines = []

model = Learner(net_config, imgc=1, imgsz=28)
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01
meta_lr = 0.0001
n_epochs = 15
n_metaepochs = 2

metaoptimizer = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.9)

for metaepoch in range(n_metaepochs):

    printlines.append('===============================')
    printlines.append(f'//           Meta-Epoch {metaepoch + 1}       //')    
    printlines.append('===============================')
    print('===============================')
    print(f'//           Meta-Epoch {metaepoch + 1}       //')    
    print('===============================')

    for mi, metabatch in enumerate(metatrain_loader, 0):  #  Meta-step
        print(mi)
        printlines.append(f'{mi} updates at Meta-Level')
        print(f'{mi} updates at Meta-Level')

        running_loss = 0.0  #  At each meta-step, the loss is reset

        # No need to store initial weights

        for pi, problem_loaders in enumerate(metabatch, 0):  #  Problem in the meta-batch

            printlines.append(f'- Problem {pi + 1} -')
            print(f'- Problem {pi + 1} -')

            problem_loader = problem_loaders['train']
            problem_loader_val = problem_loaders['val']
            ref_label = None

            new_weights = model.parameters()

            for epoch in range(n_epochs):  #  Epoch in the problem training

                printlines.append(f'Epoch {epoch + 1}')
                print(f'Epoch {epoch + 1}')

                val_loss = 0.0
                val_accuracy = 0.0

                for i, data in enumerate(problem_loader, 0):  #  Step in the problem

                    inputs_raw, labels_raw = data
                    inputs = 1 - inputs_raw
                    outputs = model(inputs, new_weights)
                    if ref_label is None:
                        ref_label = labels_raw[0]   #  On a new problem (1st step) adjust label mapping
                    labels = process_labels(labels_raw, ref_label)

                    new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                    #  As the prediction is intrinsically done with the new weights, no need to actually update the model at the Learning Level

                    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
                    print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation

                    inputs_rawv, labels_rawv = datav
                    inputsv = 1 - inputs_rawv
                    outputsv = model(inputsv, new_weights)
                    labelsv = process_labels(labels_rawv, ref_label)

                    lossv = criterion(outputsv, labelsv)  #  Loss in a validation batch
                    val_loss += lossv.item()
                    val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                printlines.append(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation
                print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation

            running_loss += lossv  #  After all epochs (all training process) in a single problem the validation loss is added

            # Again, no need to update the model to the initial weights 
        
        metastep_loss = running_loss / metabatch_size  #  The added validation losses of all problems in the metabatch are averaged

        metaoptimizer.zero_grad()  #  We perform gradient descent at the Meta-Level over the averaged validation loss
        metastep_loss.backward()
        metaoptimizer.step()

        if (mi + 1) % 1000 == 0:  #  Meta-validation performed every 1000 meta-steps

            printlines.append('META-VALIDATION STEP:')
            print('META-VALIDATION STEP:')

            for mbvi, metabatch_val in enumerate(metaval_loader):  #  Meta-validation meta-step

                if (mbvi + 1) % 10 == 0:

                    printlines.append(f'Validation step {mbvi + 1}')
                    print(f'Validation step {mbvi + 1}')

                for problem_loaders in metabatch_val:  #  Problem in the meta-validation meta-batch

                    problem_loader = problem_loaders['train']
                    problem_loader_val = problem_loaders['val']
                    ref_label = None
                    new_weights = model.parameters()

                    for epoch in range(n_epochs):  #  Epoch in the problem training

                        val_loss = 0.0
                        val_accuracy = 0.0

                        for i, data in enumerate(problem_loader, 0):  #  Step in the problem
                            
                            inputs_raw, labels_raw = data
                            inputs = 1 - inputs_raw
                            outputs = model(inputs)
                            if ref_label is None:
                                ref_label = labels_raw[0]
                            labels = process_labels(labels_raw, ref_label)

                            new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                        #    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                        for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation, as in Meta-Train

                            inputs_rawv, labels_rawv = datav
                            inputsv = 1 - inputs_rawv
                            outputsv = model(inputsv)
                            labelsv = process_labels(labels_rawv, ref_label)
                            
                            lossv = criterion(outputsv, labelsv)
                            val_loss += lossv.item()
                            val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                    
                    if (mbvi + 1) % 10 == 0:

                        printlines.append(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)
                        print(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)

            printlines.append('END OF META-VALIDATION STEP')
            print('END OF META-VALIDATION STEP')





```

    ... 



```python
with open('metalearning_002.txt', 'w') as f:
    for line in printlines:
        f.write(f"{line}\n")
```


```python
meta_learning_training_output_lines = open('metalearning_002.txt', 'r').readlines()
```


```python
metalearning_training = []
crt_metaepoch = 0
crt_metastep = 0
crt_problem = 0
new_metaepoch = False
new_metastep = False
on_metaval = False

for line in meta_learning_training_output_lines:
    print(line)
    if 'Meta-Epoch' in line:
        metaepoch_line_core = line.strip('//')
        metaepoch = int([word for word in metaepoch_line_core.split(' ') if word][1])
        assert metaepoch == (crt_metaepoch + 1)
        crt_metaepoch = metaepoch
        new_metaepoch = True
    elif ' updates at Meta-Level' in line and line.split(' ')[0].isdigit():
        metastep = int(line.split(' ')[0]) + 1
        assert (metastep == (crt_metastep + 1)) or (new_metaepoch and metastep == 1)
        crt_metastep = metastep
        metalearning_training.append({'Meta-Epoch': crt_metaepoch, 
                                      'Meta-step': crt_metastep,
                                      'Problems' : []})
        new_metastep = True
        new_metaepoch = False
    elif 'Problem' in line and line.split(' ')[1] == 'Problem' and line.split(' ')[2].isdigit():
        problem_num = int(line.split(' ')[2])
        assert (problem_num == (crt_problem + 1)) or (new_metastep and problem_num == 1)
        metalearning_training[-1]['Problems'].append([])
        crt_problem = problem_num
        crt_epoch = 0
        crt_step = 0
        new_epoch = False
    elif line.startswith('Epoch '):
        if all([chr.isdigit() for chr in line[6:-1]]):
            crt_epoch = int(line[6:-1])
            new_epoch = True
        else:            
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            epoch = int(line_epoch_str.split(' ')[1])
            assert epoch == crt_epoch
            assert line_loss_str.split(': ')[0] == 'Loss'
            assert line_accuracy_str.split(': ')[0] == 'Accuracy'
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            if line_step_str.strip(']') == 'VALIDATION':
                metalearning_training[-1]['Problems'][-1][-1]['validation'] = \
                    {'loss': loss_value, 'accuracy': accuracy_value}
            else:
                step = int(line_step_str.strip(']').split(' ')[-1])
                assert (step == (crt_step + 1)) or (new_epoch and step == 1)
                crt_step = step
                metalearning_training[-1]['Problems'][-1].append(
                    {'epoch': epoch, 'step': step, 'train': {'loss': loss_value, 
                                                   'accuracy': accuracy_value}})
            new_epoch = False
    elif 'META-VALIDATION STEP:' in line:
        metalearning_training[-1]['Meta-Validation'] = []
        on_metaval = True
    elif line.startswith('Validation step'):
        assert on_metaval
        valstep = int(line.split(' ')[-1])
        metalearning_training[-1]['Meta-Validation'].append(
            {'step': valstep, 'problems': []})
    elif line.startswith('Last epoch, VALIDATION'):
        if on_metaval:
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            metalearning_training[-1]['Meta-Validation'][-1]['problems'].append(
                {'loss': loss_value, 'accuracy': accuracy_value})
        else:
            continue
    elif 'END OF META-VALIDATION STEP' in line:
        on_metaval = False
    else:
        continue



```

    ...
    



```python
plt.plot([pstep['train']['loss'] for pstep in metalearning_training[0]['Problems'][0]])
```




    [<matplotlib.lines.Line2D at 0x7f07de7b08e0>]




    
![Imgur](https://i.imgur.com/dVREBzT.png)
    



```python
metalearning_training = metalearning_training[:-1]
```


```python
plt.figure(figsize=(25,50))
vertical = int(len(metalearning_training) / 100) + 1
for iims in range(vertical):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(vertical, 8, (iims * 8) + iprb + 1)
        plt.plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]])
```


    
![Imgur](https://i.imgur.com/2u9TTx0.png)
    



```python
plt.figure(figsize=(25,50))
vertical = int(len(metalearning_training) / 100) + 1
for iims in range(vertical):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(vertical, 8, (iims * 8) + iprb + 1)
        plt.plot(smooth_plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]], 5))
```


    
![Imgur](https://i.imgur.com/durDGE6.png)
    



```python
plt.plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f0810a672e0>]




    
![Imgur](https://i.imgur.com/qegOjaS.png)
    



```python
plt.plot(smooth_plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f080c271b80>]




    
![Imgur](https://i.imgur.com/gnIkQ71.png)
    



```python
plt.figure(figsize=(25,200))
for ims, metastep in enumerate(metalearning_training):
    if ims % 10 != 0:
        continue
    for iprb in range(8):
        plt.subplot(130, 8, (int(ims / 10) * 8) + iprb + 1)
        plt.plot([prbstep['validation']['loss'] for prbstep in metastep['Problems'][iprb] if 'validation' in prbstep.keys()])

```


    
![Imgur](https://i.imgur.com/7nRMQl4.png)
    



```python
plt.plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f080c1b90d0>]




    
![Imgur](https://i.imgur.com/V8szfgb.png)
    


And the smoothed version


```python
plt.plot(smooth_plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f0807986490>]




    
![Imgur](https://i.imgur.com/pWgUSvo.png)
    


We clearly see a more stable tendency on the Meta-Train Loss. However, it seems to work until the ~650th step and then raise again. Probably the behaviour of the LR in SGD (even using momentum) limits the whole capacity of the model. We may try to use other (meta-)optimizers.

#### Adam as Meta-optimizer

There is [several literature](https://opt-ml.org/papers/2021/paper53.pdf) on how behaves Adam in contrast with SGD that I will not explain here since it should require another whole post. Just note that by using Adam we will switch to the Meta-optimizer that the original MAML authors use (as well as the one used in the unofficial Pytorch implementation where we took the Learner class from). Adam will adjust the LR somehow automatically and tends to be less sensitive to the initial LR than SGD.


```python
printlines = []

model = Learner(net_config, imgc=1, imgsz=28)
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01
meta_lr = 0.0001
n_epochs = 15
n_metaepochs = 2

metaoptimizer = optim.Adam(model.parameters(), lr=meta_lr)

for metaepoch in range(n_metaepochs):

    printlines.append('===============================')
    printlines.append(f'//           Meta-Epoch {metaepoch + 1}       //')    
    printlines.append('===============================')
    print('===============================')
    print(f'//           Meta-Epoch {metaepoch + 1}       //')    
    print('===============================')

    for mi, metabatch in enumerate(metatrain_loader, 0):  #  Meta-step
        print(mi)
        printlines.append(f'{mi} updates at Meta-Level')
        print(f'{mi} updates at Meta-Level')

        running_loss = 0.0  #  At each meta-step, the loss is reset

        # No need to store initial weights

        for pi, problem_loaders in enumerate(metabatch, 0):  #  Problem in the meta-batch

            printlines.append(f'- Problem {pi + 1} -')
            print(f'- Problem {pi + 1} -')

            problem_loader = problem_loaders['train']
            problem_loader_val = problem_loaders['val']
            ref_label = None

            new_weights = model.parameters()

            for epoch in range(n_epochs):  #  Epoch in the problem training

                printlines.append(f'Epoch {epoch + 1}')
                print(f'Epoch {epoch + 1}')

                val_loss = 0.0
                val_accuracy = 0.0

                for i, data in enumerate(problem_loader, 0):  #  Step in the problem

                    inputs_raw, labels_raw = data
                    inputs = 1 - inputs_raw
                    outputs = model(inputs, new_weights)
                    if ref_label is None:
                        ref_label = labels_raw[0]   #  On a new problem (1st step) adjust label mapping
                    labels = process_labels(labels_raw, ref_label)

                    new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                    #  As the prediction is intrinsically done with the new weights, no need to actually update the model at the Learning Level

                    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
                    print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation

                    inputs_rawv, labels_rawv = datav
                    inputsv = 1 - inputs_rawv
                    outputsv = model(inputsv, new_weights)
                    labelsv = process_labels(labels_rawv, ref_label)

                    lossv = criterion(outputsv, labelsv)  #  Loss in a validation batch
                    val_loss += lossv.item()
                    val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                printlines.append(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation
                print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation

            running_loss += lossv  #  After all epochs (all training process) in a single problem the validation loss is added

            # Again, no need to update the model to the initial weights 
        
        metastep_loss = running_loss / metabatch_size  #  The added validation losses of all problems in the metabatch are averaged

        metaoptimizer.zero_grad()  #  We perform gradient descent at the Meta-Level over the averaged validation loss
        metastep_loss.backward()
        metaoptimizer.step()

        if (mi + 1) % 1000 == 0:  #  Meta-validation performed every 1000 meta-steps

            printlines.append('META-VALIDATION STEP:')
            print('META-VALIDATION STEP:')

            for mbvi, metabatch_val in enumerate(metaval_loader):  #  Meta-validation meta-step

                if (mbvi + 1) % 10 == 0:

                    printlines.append(f'Validation step {mbvi + 1}')
                    print(f'Validation step {mbvi + 1}')

                for problem_loaders in metabatch_val:  #  Problem in the meta-validation meta-batch

                    problem_loader = problem_loaders['train']
                    problem_loader_val = problem_loaders['val']
                    ref_label = None
                    new_weights = model.parameters()

                    for epoch in range(n_epochs):  #  Epoch in the problem training

                        val_loss = 0.0
                        val_accuracy = 0.0

                        for i, data in enumerate(problem_loader, 0):  #  Step in the problem
                            
                            inputs_raw, labels_raw = data
                            inputs = 1 - inputs_raw
                            outputs = model(inputs)
                            if ref_label is None:
                                ref_label = labels_raw[0]
                            labels = process_labels(labels_raw, ref_label)

                            new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                        #    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                        for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation, as in Meta-Train

                            inputs_rawv, labels_rawv = datav
                            inputsv = 1 - inputs_rawv
                            outputsv = model(inputsv)
                            labelsv = process_labels(labels_rawv, ref_label)
                            
                            lossv = criterion(outputsv, labelsv)
                            val_loss += lossv.item()
                            val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                    
                    if (mbvi + 1) % 10 == 0:

                        printlines.append(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)
                        print(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)

            printlines.append('END OF META-VALIDATION STEP')
            print('END OF META-VALIDATION STEP')





```

    ...



```python
with open('metalearning_003.txt', 'w') as f:
    for line in printlines:
        f.write(f"{line}\n")
```


```python
meta_learning_training_output_lines = open('metalearning_003.txt', 'r').readlines()
```


```python
metalearning_training = []
crt_metaepoch = 0
crt_metastep = 0
crt_problem = 0
new_metaepoch = False
new_metastep = False
on_metaval = False

for line in meta_learning_training_output_lines:
    print(line)
    if 'Meta-Epoch' in line:
        metaepoch_line_core = line.strip('//')
        metaepoch = int([word for word in metaepoch_line_core.split(' ') if word][1])
        assert metaepoch == (crt_metaepoch + 1)
        crt_metaepoch = metaepoch
        new_metaepoch = True
    elif ' updates at Meta-Level' in line and line.split(' ')[0].isdigit():
        metastep = int(line.split(' ')[0]) + 1
        assert (metastep == (crt_metastep + 1)) or (new_metaepoch and metastep == 1)
        crt_metastep = metastep
        metalearning_training.append({'Meta-Epoch': crt_metaepoch, 
                                      'Meta-step': crt_metastep,
                                      'Problems' : []})
        new_metastep = True
        new_metaepoch = False
    elif 'Problem' in line and line.split(' ')[1] == 'Problem' and line.split(' ')[2].isdigit():
        problem_num = int(line.split(' ')[2])
        assert (problem_num == (crt_problem + 1)) or (new_metastep and problem_num == 1)
        metalearning_training[-1]['Problems'].append([])
        crt_problem = problem_num
        crt_epoch = 0
        crt_step = 0
        new_epoch = False
    elif line.startswith('Epoch '):
        if all([chr.isdigit() for chr in line[6:-1]]):
            crt_epoch = int(line[6:-1])
            new_epoch = True
        else:            
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            epoch = int(line_epoch_str.split(' ')[1])
            assert epoch == crt_epoch
            assert line_loss_str.split(': ')[0] == 'Loss'
            assert line_accuracy_str.split(': ')[0] == 'Accuracy'
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            if line_step_str.strip(']') == 'VALIDATION':
                metalearning_training[-1]['Problems'][-1][-1]['validation'] = \
                    {'loss': loss_value, 'accuracy': accuracy_value}
            else:
                step = int(line_step_str.strip(']').split(' ')[-1])
                assert (step == (crt_step + 1)) or (new_epoch and step == 1)
                crt_step = step
                metalearning_training[-1]['Problems'][-1].append(
                    {'epoch': epoch, 'step': step, 'train': {'loss': loss_value, 
                                                   'accuracy': accuracy_value}})
            new_epoch = False
    elif 'META-VALIDATION STEP:' in line:
        metalearning_training[-1]['Meta-Validation'] = []
        on_metaval = True
    elif line.startswith('Validation step'):
        assert on_metaval
        valstep = int(line.split(' ')[-1])
        metalearning_training[-1]['Meta-Validation'].append(
            {'step': valstep, 'problems': []})
    elif line.startswith('Last epoch, VALIDATION'):
        if on_metaval:
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            metalearning_training[-1]['Meta-Validation'][-1]['problems'].append(
                {'loss': loss_value, 'accuracy': accuracy_value})
        else:
            continue
    elif 'END OF META-VALIDATION STEP' in line:
        on_metaval = False
    else:
        continue



```

    ...
    



```python
plt.plot([pstep['train']['loss'] for pstep in metalearning_training[0]['Problems'][0]])
```




    [<matplotlib.lines.Line2D at 0x7f0807688f40>]




    
![Imgur](https://i.imgur.com/GKwU7My.png)
    



```python
metalearning_training = metalearning_training[:-1]
```


```python
plt.figure(figsize=(25,50))
vertical = int(len(metalearning_training) / 100) + 1
for iims in range(vertical):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(vertical, 8, (iims * 8) + iprb + 1)
        plt.plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]])
```


    
![Imgur](https://i.imgur.com/IDMvVpp.png)
    



```python
plt.figure(figsize=(25,50))
vertical = int(len(metalearning_training) / 100) + 1
for iims in range(vertical):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(vertical, 8, (iims * 8) + iprb + 1)
        plt.plot(smooth_plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]], 5))
```


    
![Imgur](https://i.imgur.com/sIWcs6k.png)
    



```python
plt.plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f07de7927c0>]




    
![Imgur](https://i.imgur.com/dUb9WFl.png)
    



```python
plt.plot(smooth_plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f07db5c7b20>]




    
![Imgur](https://i.imgur.com/eYVQNWb.png)
    



```python
plt.figure(figsize=(25,200))
for ims, metastep in enumerate(metalearning_training):
    if ims % 10 != 0:
        continue
    for iprb in range(8):
        plt.subplot(130, 8, (int(ims / 10) * 8) + iprb + 1)
        plt.plot([prbstep['validation']['loss'] for prbstep in metastep['Problems'][iprb] if 'validation' in prbstep.keys()])

```


    
![Imgur](https://i.imgur.com/9D27LTQ.png)
    



```python
plt.plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f0815472970>]




    
![Imgur](https://i.imgur.com/uV7ujhU.png)
    


And the smoothed version


```python
plt.plot(smooth_plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f0811a489a0>]




    
![Imgur](https://i.imgur.com/uHN8k2R.png)
    


But here again, the Meta-training is unstable and the tendency is to raise the Meta-Loss. Note that we used the Meta-LR from the previous experiment, but maybe we should use an even more reduced one.

#### Adam with even lower Meta-LR


```python
printlines = []

model = Learner(net_config, imgc=1, imgsz=28)
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01
meta_lr = 0.00001
n_epochs = 15
n_metaepochs = 2

metaoptimizer = optim.Adam(model.parameters(), lr=meta_lr)

for metaepoch in range(n_metaepochs):

    printlines.append('===============================')
    printlines.append(f'//           Meta-Epoch {metaepoch + 1}       //')    
    printlines.append('===============================')
    print('===============================')
    print(f'//           Meta-Epoch {metaepoch + 1}       //')    
    print('===============================')

    for mi, metabatch in enumerate(metatrain_loader, 0):  #  Meta-step
        print(mi)
        printlines.append(f'{mi} updates at Meta-Level')
        print(f'{mi} updates at Meta-Level')

        running_loss = 0.0  #  At each meta-step, the loss is reset

        # No need to store initial weights

        for pi, problem_loaders in enumerate(metabatch, 0):  #  Problem in the meta-batch

            printlines.append(f'- Problem {pi + 1} -')
            print(f'- Problem {pi + 1} -')

            problem_loader = problem_loaders['train']
            problem_loader_val = problem_loaders['val']
            ref_label = None

            new_weights = model.parameters()

            for epoch in range(n_epochs):  #  Epoch in the problem training

                printlines.append(f'Epoch {epoch + 1}')
                print(f'Epoch {epoch + 1}')

                val_loss = 0.0
                val_accuracy = 0.0

                for i, data in enumerate(problem_loader, 0):  #  Step in the problem

                    inputs_raw, labels_raw = data
                    inputs = 1 - inputs_raw
                    outputs = model(inputs, new_weights)
                    if ref_label is None:
                        ref_label = labels_raw[0]   #  On a new problem (1st step) adjust label mapping
                    labels = process_labels(labels_raw, ref_label)

                    new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                    #  As the prediction is intrinsically done with the new weights, no need to actually update the model at the Learning Level

                    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
                    print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation

                    inputs_rawv, labels_rawv = datav
                    inputsv = 1 - inputs_rawv
                    outputsv = model(inputsv, new_weights)
                    labelsv = process_labels(labels_rawv, ref_label)

                    lossv = criterion(outputsv, labelsv)  #  Loss in a validation batch
                    val_loss += lossv.item()
                    val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                printlines.append(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation
                print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  #  Loss and accuracy averaged for all validation batches in the problem, displayed after whole validation

            running_loss += lossv  #  After all epochs (all training process) in a single problem the validation loss is added

            # Again, no need to update the model to the initial weights 
        
        metastep_loss = running_loss / metabatch_size  #  The added validation losses of all problems in the metabatch are averaged

        metaoptimizer.zero_grad()  #  We perform gradient descent at the Meta-Level over the averaged validation loss
        metastep_loss.backward()
        metaoptimizer.step()

        if (mi + 1) % 1000 == 0:  #  Meta-validation performed every 1000 meta-steps

            printlines.append('META-VALIDATION STEP:')
            print('META-VALIDATION STEP:')

            for mbvi, metabatch_val in enumerate(metaval_loader):  #  Meta-validation meta-step

                if (mbvi + 1) % 10 == 0:

                    printlines.append(f'Validation step {mbvi + 1}')
                    print(f'Validation step {mbvi + 1}')

                for problem_loaders in metabatch_val:  #  Problem in the meta-validation meta-batch

                    problem_loader = problem_loaders['train']
                    problem_loader_val = problem_loaders['val']
                    ref_label = None
                    new_weights = model.parameters()

                    for epoch in range(n_epochs):  #  Epoch in the problem training

                        val_loss = 0.0
                        val_accuracy = 0.0

                        for i, data in enumerate(problem_loader, 0):  #  Step in the problem
                            
                            inputs_raw, labels_raw = data
                            inputs = 1 - inputs_raw
                            outputs = model(inputs)
                            if ref_label is None:
                                ref_label = labels_raw[0]
                            labels = process_labels(labels_raw, ref_label)

                            new_weights, loss, accuracy = make_step(model, outputs, labels, update_lr, new_weights)

                        #    printlines.append(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')

                        for iv, datav in enumerate(problem_loader_val):  #  At the end of the training process in an epoch of a problem we compute a whole validation, as in Meta-Train

                            inputs_rawv, labels_rawv = datav
                            inputsv = 1 - inputs_rawv
                            outputsv = model(inputsv)
                            labelsv = process_labels(labels_rawv, ref_label)
                            
                            lossv = criterion(outputsv, labelsv)
                            val_loss += lossv.item()
                            val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()

                    
                    if (mbvi + 1) % 10 == 0:

                        printlines.append(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)
                        print(f'Last epoch, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')  # The Meta-Validation only runs for informative matters, so our goal is to have this at the end of each problem (every 10 steps)

            printlines.append('END OF META-VALIDATION STEP')
            print('END OF META-VALIDATION STEP')





```

    ... 



```python
with open('metalearning_005.txt', 'w') as f:
    for line in printlines:
        f.write(f"{line}\n")
```


```python
meta_learning_training_output_lines = open('metalearning_005.txt', 'r').readlines()
```


```python
metalearning_training = []
crt_metaepoch = 0
crt_metastep = 0
crt_problem = 0
new_metaepoch = False
new_metastep = False
on_metaval = False

for line in meta_learning_training_output_lines:
    print(line)
    if 'Meta-Epoch' in line:
        metaepoch_line_core = line.strip('//')
        metaepoch = int([word for word in metaepoch_line_core.split(' ') if word][1])
        assert metaepoch == (crt_metaepoch + 1)
        crt_metaepoch = metaepoch
        new_metaepoch = True
    elif ' updates at Meta-Level' in line and line.split(' ')[0].isdigit():
        metastep = int(line.split(' ')[0]) + 1
        assert (metastep == (crt_metastep + 1)) or (new_metaepoch and metastep == 1)
        crt_metastep = metastep
        metalearning_training.append({'Meta-Epoch': crt_metaepoch, 
                                      'Meta-step': crt_metastep,
                                      'Problems' : []})
        new_metastep = True
        new_metaepoch = False
    elif 'Problem' in line and line.split(' ')[1] == 'Problem' and line.split(' ')[2].isdigit():
        problem_num = int(line.split(' ')[2])
        assert (problem_num == (crt_problem + 1)) or (new_metastep and problem_num == 1)
        metalearning_training[-1]['Problems'].append([])
        crt_problem = problem_num
        crt_epoch = 0
        crt_step = 0
        new_epoch = False
    elif line.startswith('Epoch '):
        if all([chr.isdigit() for chr in line[6:-1]]):
            crt_epoch = int(line[6:-1])
            new_epoch = True
        else:            
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            epoch = int(line_epoch_str.split(' ')[1])
            assert epoch == crt_epoch
            assert line_loss_str.split(': ')[0] == 'Loss'
            assert line_accuracy_str.split(': ')[0] == 'Accuracy'
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            if line_step_str.strip(']') == 'VALIDATION':
                metalearning_training[-1]['Problems'][-1][-1]['validation'] = \
                    {'loss': loss_value, 'accuracy': accuracy_value}
            else:
                step = int(line_step_str.strip(']').split(' ')[-1])
                assert (step == (crt_step + 1)) or (new_epoch and step == 1)
                crt_step = step
                metalearning_training[-1]['Problems'][-1].append(
                    {'epoch': epoch, 'step': step, 'train': {'loss': loss_value, 
                                                   'accuracy': accuracy_value}})
            new_epoch = False
    elif 'META-VALIDATION STEP:' in line:
        metalearning_training[-1]['Meta-Validation'] = []
        on_metaval = True
    elif line.startswith('Validation step'):
        assert on_metaval
        valstep = int(line.split(' ')[-1])
        metalearning_training[-1]['Meta-Validation'].append(
            {'step': valstep, 'problems': []})
    elif line.startswith('Last epoch, VALIDATION'):
        if on_metaval:
            line_parts = line.split(', ')
            line_epoch_str = line_parts[0]
            line_step_str = line_parts[1]
            line_loss_str = line_parts[2]
            line_accuracy_str = line_parts[3]
            loss_value = float(line_loss_str.split(': ')[1])
            accuracy_value = float(line_accuracy_str.split(': ')[1])
            metalearning_training[-1]['Meta-Validation'][-1]['problems'].append(
                {'loss': loss_value, 'accuracy': accuracy_value})
        else:
            continue
    elif 'END OF META-VALIDATION STEP' in line:
        on_metaval = False
    else:
        continue



```

    ...
    



```python
plt.plot([pstep['train']['loss'] for pstep in metalearning_training[0]['Problems'][0]])
```




    [<matplotlib.lines.Line2D at 0x7f07d87e03a0>]




    
![Imgur](https://i.imgur.com/tUvM0EO.png)
    



```python
metalearning_training = metalearning_training[:-1]
```


```python
plt.figure(figsize=(25,50))
vertical = int(len(metalearning_training) / 100) + 1
for iims in range(vertical):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(vertical, 8, (iims * 8) + iprb + 1)
        plt.plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]])
```


    
![Imgur](https://i.imgur.com/lBoGBH6.png)
    



```python
plt.figure(figsize=(25,50))
vertical = int(len(metalearning_training) / 100) + 1
for iims in range(vertical):
    ims = iims * 100
    for iprb in range(8):
        plt.subplot(vertical, 8, (iims * 8) + iprb + 1)
        plt.plot(smooth_plot([pstep['train']['loss'] for pstep in metalearning_training[ims]['Problems'][iprb]], 5))
```


    
![Imgur](https://i.imgur.com/XeAb5uP.png)
    



```python
plt.plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f081454ab50>]




    
![Imgur](https://i.imgur.com/qxlE2Zu.png)
    



```python
plt.plot(smooth_plot([sum([prob[-1]['train']['loss'] for prob in mstep['Problems']]) / 8 for mstep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f080e928fa0>]




    
![Imgur](https://i.imgur.com/35IQH2b.png)
    



```python
plt.figure(figsize=(25,200))
for ims, metastep in enumerate(metalearning_training):
    if ims % 10 != 0:
        continue
    for iprb in range(8):
        plt.subplot(130, 8, (int(ims / 10) * 8) + iprb + 1)
        plt.plot([prbstep['validation']['loss'] for prbstep in metastep['Problems'][iprb] if 'validation' in prbstep.keys()])

```


    
![Imgur](https://i.imgur.com/E9DMEbs.png)
    



```python
plt.plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training])
```




    [<matplotlib.lines.Line2D at 0x7f08106fa0a0>]




    
![Imgur](https://i.imgur.com/onvGAka.png)
    


And the smoothed version


```python
plt.plot(smooth_plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training], 50))
```




    [<matplotlib.lines.Line2D at 0x7f081c4bc910>]




    
![Imgur](https://i.imgur.com/lc8Zz6U.png)
    



```python
plt.plot(smooth_plot([(sum([problem[-1]['validation']['loss'] for problem in metastep['Problems']]) / 8) for metastep in metalearning_training], 300)[10:])
```




    [<matplotlib.lines.Line2D at 0x7f081c312ee0>]




    
![Imgur](https://i.imgur.com/ZrhqDlJ.png)
    


Ok, so in this case the overall tendency of the Meta-Train Loss is the one we were looking for. However, although the tendency is right we may notice that the improvement is not so notorious in Loss terms. Actually, the first Meta-step achieved the lowest loss.

### Considerations

Although we have probably not reached the whole potential of our Meta-Learning pipeline, we can get some ideas from there. The most imporant issue we are facing even with our best configuration (Adam with the even lower Meta-LR we tried) the contribution of Meta-Learning has even less impact than inter-(Meta-)batch variations and the Meta-Training Loss is extremely unstable. Thus, one of the main inconveniences we may be introducing is choosing a too small Meta-batch size.

However, with the resources we are using we are limited in this sense. In this post we will not follow this path. There is a workaround that could smooth some impact which could be using a [Batch Normalization](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/) technique in the Meta-Learning level (Meta-batch normalization?). Again, I will not explain Batch Normalization with all the content it contains, but for those unfamiliar it consists in scaling the input samples into a fixed mean and standard deviation. That helps (among other benefits) stabilize the training. Did you notice the issue in our case? Well, as we are proposing it at the Meta-Learning level, our samples are problems. There's no clue about what do we exactly need to scale for a problem. There's [literature about that](https://isis-data.science.uva.nl/cgmsnoek/pub/du-metanorm-iclr2021.pdf) but it is another whole topic, so again, we will just keep it as a future idea (it may be deceiving but that is what science in general is about).

Also, we have assumed lots of hyperparameters, but these could also be optimized. To do so, it would be great to prepare some hyperparameter search technique. I am pretending to do some post introducing [Weights and Biases](https://wandb.ai/site), a tool to run that, in a not so distant future.

## Some conclusions about the post

With that post we have already covered all the things I wanted to share about my knowledge on Meta-Learning. Although several incredibles approaches have been shown and the purely theoretical side of Meta-Learning is nearly covered, getting a real problem and making Meta-Learning run and give results in them is still hard. MAML is an incredible approach, but by itself it will be hard to have meaningful improvements.

How bad, right?

On the contrary. Meta-Learning is not a magical box where you input your model abd your data and *magically* it makes the model perform better in target data. It is, instead, a way of understanding the data. A way of wisely building training pipelines that may avoid some issues for us at certain stages.

If you remind, in my first post I didn't talk about Learning without data. I always talked about approximating the regular training process without much data. I talked about using few data to learn more optimally in a given problem. And anticipating this problem will always be a good measure. Maybe lowering the loss a couple of points is not important at all. But maybe being able to advance some preliminar results a week before getting the whole data collection just using some demo data is important. Or maybe not. Bt that depends on the scenario. And that is why ML engineers are in the teams, to try approaches and keep or discard them depending on the results, right?

And from the research point of view, the door is still open. The vanilla approaches may have been proposed, but more granular/nicho scenarios are still demanding improvements. There are pipelines (probably combining and/or modifying existing Meta-Learning approaches) are still to be discovered.

But that was an introduction. And if yoy have read until here and fought and thought with the issues we have faced in this journey, you are more than prepared to go deeper on this topic.
