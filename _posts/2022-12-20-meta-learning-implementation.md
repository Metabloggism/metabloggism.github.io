# Meta-Learning: Experiment & MAML implementation by Metabloggism

Welcome to this first Notebook by Metabloggism! This episode follows directly the post [Meta-Learning explained](https://metabloggism.github.io/2022/11/21/meta-learning.html). As a recap from it, we reviewed the meaning of Meta-Learning, how it did appear and evolve and how is it approached nowadays. We concluded that today, we use Meta-Learning mostly as a tool against scenarios with few data in a specific problem we want to solve but where we are able to find data from related problems, thus building a domain of problems in a higher level from Learning called *Meta-Learning*. At this level you may learn how to learn efficiently in each problem, and that may be done in different ways. One of the most praised approaches in the last years, and the one which probably is more commonly used in these settings is [MAML](https://arxiv.org/pdf/1703.03400.pdf), which at the Meta-Learning level learns a proper general (in the problems domain) initialization of any (learnable by gradient descent) model at the Learning level. If you still have doubts, I would recommend reading the mentioned post again before reading this Notebook.



The scope of this post is to materialize what we described in the previous one. First, we will take a tour in the experiment setting, defining what we want to solve in the end and building the scenario in which we will work on. Second, we will implement a MAML approach and train at Meta-Learning.

This is a simple walkthrough of this process. We do not aim to engage in a performance analysis since we will let this for a future post, with other variables deeply explored, a fair comparison with other approaches, with a proper hyperparameter exploration, etc.

So that said, let's engage into work!

## Preliminaries: Meta-Learning datasets

I need to introduce something I forgot in the previous post. We did not talk about Meta-Learning datasets!

### Requirements of a good Meta-Learning dataset

With all the previous theory, our definition of Meta-Learning could be something like *Learning tools that will allow to Learn to solve problems from a certain domain* and we do so by repeatedly learning to solve these problems. Thus, any dataset which allows us to do so can be considered a Meta-Learning dataset.

Meta-Learning datasets may be used to directly solve target problems, but since they don't always allow this, they are commonly used for making experiments with approaches for analysis.

In general, any usual dataset can be used as a Meta-Learning dataset. Just think about [Imagenet](https://www.image-net.org/), for example (let's think about the classification task). There you have a series of samples and labels belonging to a group of classes. There you can take the whole domain of classes and build a domain of problems consisting in binary classification problems, where you may have as many problems as combinations of 2 classes, and therefore Learn to Learn (Meta-Learn) to solve any binary classification problem there. Even more extreme, think a simple dataset that a little Startup company which makes supervised face identification handles, consisting in faces of both people in the group to identify and out from it (labeled). One could make artificial groups of the other people and build several problems of the same nature.

However, you may have note that these domains are forced and do not have a strong semantical relation, making the information extraction weak among problems, and almost not improving the performance in the target Learning one. There are some aspects that determine if a dataset is proper for Meta-Learning, which enable to exploit their relation to skip some steps in any target problem from the domain. Thus, in general any dataset can act as a Meta-Learning dataset, but not all of them will be able to become the desired Meta-learning tool. So, which ingredients must it contain?

* The ingredients of any Learning problem, e.g. for Supervised Classification problems, samples and labels.
* Labels should preferably be variated, to allow us to build a properly extense schedule at the Meta-Learning level. Note that many classes can be converted to many problems. At least, should allow to have more than one problem.
* A rich hierarchy/meta-information which should allow us to relate the instances and build meaningful domains

Moreover, commonly the Meta-Learning datasets have some common traits:

* As they aim to pose some challenge to you, they mostly have a limited number of samples per class to emulate a Few-Shot Learning scenario. In case they don't you can always enforce this limitation.
* Since they need lots of classes (and therefore lots of samples), they tend to work in light data (e.g. in Computer Vision in low-resolution).

### Datasets

#### [Omniglot](https://omniglot.com/)

One of the most common datasets. [Presented by Lake et al. in 2015](https://cims.nyu.edu/~brenden/papers/LakeEtAl2011CogSci.pdf) as a challenge to solve a Few-Shot Learning problem (although they didn't mention the term *Few-Shot*), it was the dataset chosen by Koch et al. to describe a One-Shot Learning problem (remind [the section *The comeback of Meta-Learning and its relation to Few-Shot Learning* in the previous post](https://metabloggism.github.io/2022/11/21/meta-learning.html)) and since then a benchmark in Meta-Learning. It has been used in most of the main Meta-Learning approaches, such as [MANN](), [MatchingNets](https://arxiv.org/pdf/1606.04080.pdf), [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf), [MAML](https://arxiv.org/pdf/1703.03400.pdf) or [Modular Meta-Learning](https://arxiv.org/pdf/1909.05557.pdf). An API for the dataset is also included in some top ML frameworks like Pytorch (in the torchvision package).

It consists in a series of alphabets where in each alphabet there is a series of characters and for each character a limited group of samples (20 samples). A sample cosists in a 105x105 (single-channel) image representing the character. The label of the sample is the character itself (related to an alphabet). This allows to build semantically meaningful domains of problems, depending in the alphabet where the characters belong to.

#### Mini-Imagenet

Introduced in the paper of [MatchingNets](https://arxiv.org/pdf/1606.04080v2.pdf). [It](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) is probably the second benchmark for Meta-Learning. A modified version of the [ImageNet](https://www.image-net.org/) Computer Vision dataset where a semantic selection has been performed. It is also used in some of the main approaches of Meta-Learning (aside of MatchingNets). 

#### Imagenet and other common general datasets used in Meta-Learning

Most times, the Meta-Learning approaches are used in general datasets with enforced conditions. An example is found in [Learning to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf), where authors used common datasets like [MNIST](https://yann.lecun.com/exdb/mnist/) or [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

#### Non-CV datasets

Note that we only considered CV datasets until now. I come from the CV field, but I do not pretend to focus in CV in this episode. However, these datasetys are actually the ones mostly used when testing the main Meta-Learning approaches. Alternatively, some regression tasks are also tested like in [MAML](https://arxiv.org/pdf/1703.03400.pdf). 

## Experiment definition

There are two main components we may want to implement in the notebook: the challenge and the solution.

We will work in the Omniglot dataset since it is the most standard benchmark as well as it provides a hierarchy we will use to build the domain.

The challenge I propose to solve is *Learning to be able to solve, given a random alphabet, a character (within the alphabet) binary classification task with few samples*. Thus, our setting will go as follows:



*   Build all the possible problems of this type in Omniglot: 

    * Pick each combinations of 2 characters within all alphabets

    * For each combination, get all the samples and make common ML splits: train, validation and test

*  Make splits of alphabets at Meta-Learning level: Meta-train, Meta-validation and Meta-test:
    * For each Meta-split (Meta-sets from now on) take all possible problems for all its alphabets

We aim to be able to take any of the Meta-test problems and after a proper training of that problem, be able to solve its test set.

We rely in the fact that the Meta-sets will be enough varied (they will be randomly sorted) to represent the whole domain each.

As for the solution, we will use the MAML approach. Remember that this approach works by, at each meta-step, take a meta-batch of problems and for each problem compute an individual training (over its training samples) and evaluate the loss in its test set, averaging it over all the problems in the meta-batch and thus resulting in the meta-step loss.

I found in Github [the official implementation](https://github.com/cbfinn/maml) of MAML by Chelsea Finn, in Tensorflow, as well as other unofficial implementations in Pytorch, like [this one](https://github.com/dragen1860/MAML-Pytorch). However, we want to take control of the implementations to bring it to our own experiment, so we will develop it from scratch (but following the ideas there).



## Tools

For this Proof of Concept we will use Python and Pytorch as our ML framework.

I developed it in a Google Collab so I may be able to use GPU's.

## Code

### Imports and setting

The following cell can be skipped since it is just necessary for running the Google Collab Notebook in my Drive environment.


```python
from google.colab import drive
drive.mount('/content/drive')   
# WRITE PATH WHERE YOU WERE
%cd drive/MyDrive/collab_space/metabloggism/meta-learning
```

    Mounted at /content/drive
    /content/drive/MyDrive/collab_space/metabloggism/meta-learning


Imports, we'll skip explanations, I just came back here to import any module that we needed.


```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, BatchSampler
import torchvision
import matplotlib.pyplot as plt
```

### Exploring torchvision's Omniglot dataset

As mentioned before, torchvision offers an API for handling Omniglot. We'll dissect its composition now.

First, we must initialize it following torchvision's Omniglot [documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.Omniglot.html). We'll download the whole ZIP of images(*download=True* and specifying a *root* directory, if you don't want it just set *download* to *False) and the only transform we want is having it as tensors for our future NN. 


```python
dataset = torchvision.datasets.Omniglot(
    root="./dataset/omniglot", download=True, transform=torchvision.transforms.ToTensor()
)
```

    Files already downloaded and verified


Ok, with that loaded, let's check the object's appearance.


```python
dataset
```




    Dataset Omniglot
        Number of datapoints: 19280
        Root location: ./dataset/omniglot/omniglot-py
        StandardTransform
    Transform: ToTensor()



Everything seems just as we asked, with a dataset of ~19k data points.

Exploraing a bit the documentation, we'll see that the only way to use the dataset is [through its *\__getitem__*](https://pytorch.org/vision/stable/generated/torchvision.datasets.Omniglot.html#torchvision.datasets.Omniglot.__getitem__), i.e. you can only get any of its elements (from its ~19k), so let's get for example the first one and review it.


```python
dataset[0]
```




    (tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
              [1., 1., 1.,  ..., 1., 1., 1.],
              [1., 1., 1.,  ..., 1., 1., 1.],
              ...,
              [1., 1., 1.,  ..., 1., 1., 1.],
              [1., 1., 1.,  ..., 1., 1., 1.],
              [1., 1., 1.,  ..., 1., 1., 1.]]]), 0)



Ok, it seems to be a tuple as the documentation says. According to it, the first element is the image itself (a torch tensor) while the second one is the target label). Let's review both


```python
dataset[0][0].shape
```




    torch.Size([1, 105, 105])



As expected, the image is a single image one of 105x105. Let's plot its only channel.


```python
plt.imshow(dataset[0][0][0].numpy())
```




    <matplotlib.image.AxesImage at 0x7f1fca5129d0>




    
![png](Metabloggism_Meta-Learning_files/Metabloggism_Meta-Learning_36_1.png)
    


So this is the appearance of an Omniglot symbol. Everything seems find with the samples then, and we know how to get them. Let's go for the labels.

We may review the first 100 labels.


```python
[dataset[ismp][1] for ismp in range(100)]
```




    [0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     0,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     1,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     3,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4]



Do you realize the pattern? Labels are sequential and each label repeats for 20 samples. Which is the number of samples per character (class). To verify that, if we get a label every 2k samples, each label should add 100 to the previous.


```python
[dataset[ismp * 2000][1] for ismp in range(8)]
```




    [0, 100, 200, 300, 400, 500, 600, 700]



Which happens.

Ok, one last thing that will be useful for us. Going deeper, appart from the documentation we can also get to the [source code](https://pytorch.org/vision/stable/_modules/torchvision/datasets/omniglot.html#Omniglot) and realize 2 useful attributes of the dataset object called *_alphabets* and *_characters*, which is the ordered list of each of them. This way, we'll have semantical information of each of the labels and therefore we may be able to build our scenario.


```python
alphabets = dataset._alphabets
```


```python
characters = dataset._characters
```

Let's check how many alphabets and characters are there.


```python
len(alphabets)
```




    30




```python
len(characters)
```




    964



Which makes sense.

The last thing, let's verify we can embed the dataset into a torch DataLoader.


```python
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)
```

    /usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py:554: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(


### Load raw dataset

So with averything that we learned, let's load our dataset and its related info.


```python
omniglot_raw = torchvision.datasets.Omniglot(root="./dataset/omniglot", download=True, transform=torchvision.transforms.ToTensor())
```

    Files already downloaded and verified



```python
alphabets = omniglot_raw._alphabets
characters = omniglot_raw._characters
```


```python
num_alphabets = len(alphabets)
num_characters = len(characters)
```

### Building the Meta-Splits

Ok, at this point we want to build each Meta-Split (or Meta-set). The material needs of a Meta-set are the alphabets it will contain. Remember that each Meta-set will determine its problems, but that is interpretablñe from its alphabets, since the problems will be all the possible problems of binary classification between characters of a same aphabet within the Meta-set.

We want to balance the Meta-set at character level, in a way in which each Meta-set will contain an approximation to a fixed ratio of characters, but respecting the constraint in which each whole alphabet must belong to a single Meta-set. 

The simples way to do so is to take each empty Meta-set and add alphabets until the fixed ratio is reached, then the last Meta-set will get the remaining alphabets. This last Meta-set will be Meta-test, since lossing a bit of its content doesn't seem a disaster aside of its statistical significance.

So first we will define a *MetaSplit* class that will contain this information. It will also contain some informative attributes such as the minimum number of characters, which will be computed at initialization depending on the Meta-set ratio and the total number of characters (these variables will be passed), the current number of characters (which will be updated out from the class when alphabets are added) and the number of problems, which will also be added from out of the class.


```python
class MetaSplit:
  def __init__(self, ratio, total_num_characters):
    self.alphabets = []
    self.num_characters = 0
    self.min_num_characters = total_num_characters * ratio
    self.num_problems = None

```

And let's initialize the Meta-splits (after this step they will still be empty)


```python
metasplits = {'metatrain': MetaSplit(0.7, num_characters),
              'metaval': MetaSplit(0.15, num_characters),
              'metatest': MetaSplit(0.15, num_characters)}
```

The following step is counting the number of character in each alphabet. To do so, we need to parse the strings of the alphabets and the characters, which go as the following.


```python
print(alphabets[0])
```

    Alphabet_of_the_Magi



```python
print(characters[0])
```

    Alphabet_of_the_Magi/character01


Ok, so easily we can see that a character goes as *alphabet/character_in_alphabet*. So, we will for each alphabet count in how many characters does it match the substring before the */*.


```python
chars_per_alphabet = {alph: [char.split('/')[0] for char in characters].count(alph) for alph in alphabets}
```


```python
chars_per_alphabet
```




    {'Alphabet_of_the_Magi': 20,
     'Anglo-Saxon_Futhorc': 29,
     'Arcadian': 26,
     'Armenian': 41,
     'Asomtavruli_(Georgian)': 40,
     'Balinese': 24,
     'Bengali': 46,
     'Blackfoot_(Canadian_Aboriginal_Syllabics)': 14,
     'Braille': 26,
     'Burmese_(Myanmar)': 34,
     'Cyrillic': 33,
     'Early_Aramaic': 22,
     'Futurama': 26,
     'Grantha': 43,
     'Greek': 24,
     'Gujarati': 48,
     'Hebrew': 22,
     'Inuktitut_(Canadian_Aboriginal_Syllabics)': 16,
     'Japanese_(hiragana)': 52,
     'Japanese_(katakana)': 47,
     'Korean': 40,
     'Latin': 26,
     'Malay_(Jawi_-_Arabic)': 40,
     'Mkhedruli_(Georgian)': 41,
     'N_Ko': 33,
     'Ojibwe_(Canadian_Aboriginal_Syllabics)': 14,
     'Sanskrit': 42,
     'Syriac_(Estrangelo)': 23,
     'Tagalog': 17,
     'Tifinagh': 55}



And finally let's shuffle the alphabets and split among the Meta-sets!


```python
random.shuffle(alphabets)
```

As said, we will take each alphabet and add it to its corresponding Meta-set. To do so, we will begin with a given Meta-set (in our case the order is Meta-train, Meta-validation and Meta-test) and check if its internal number of characters has reached its minimum value so in case it does we will switch to the next Meta-set. In any case we will add the current alphabet to the current Meta-set and update the number of characters.


```python
current_metasplit = 'metatrain'
switch_metasplit_from = {'metatrain': 'metaval', 'metaval': 'metatest'}

for alphabet in alphabets:
  if not metasplits[current_metasplit].num_characters < metasplits[current_metasplit].min_num_characters:
    current_metasplit = switch_metasplit_from[current_metasplit]
  metasplits[current_metasplit].alphabets.append(alphabet)
  metasplits[current_metasplit].num_characters += chars_per_alphabet[alphabet]

```

We still have to compute the number of problems of each Meta-set, which depennds on its alphabets (and the number of characters in each of these alphabets).

I developed a formula to compute this which goes as follows:

* As each problem's characters must be of the same alphabet, we can not group all characters in a single pool. Instead, we will compute it alphabet-wise and sum for all its alpabets. If we call $P$ the number of problems, $ P = \sum_{\alpha \in MS}{P_{\alpha}}$, where $P_{\alpha}$ is the number of problems in the alphabet $\alpha$ and $MS$ is the Meta-set (so the number of problems is the sum of the number of problems in each alphabet). This information is available right now (in the *MetaSplit* objects and the *chars_per_alphabet* variable).

* The number of problems in an alphabet is the number of possible combinations of two characters among all belonging to the alphabet. From statistical theory, that is a Combination without repetition with combinations of 2, i.e. if $C_{\alpha}$ is the number of characters in the alphabet $\alpha$, ${C_{\alpha} \choose 2}$. We also know that ${m \choose n}$ is computed as $\frac{m!}{n! (m - n)!}$, so in our case ${C_{\alpha} \choose 2} = \frac{C_{\alpha}!}{2! (C_{\alpha} - 2)!}$.

* We know that $2! = 2$. We also know that $\frac{C_{\alpha}!}{(C_{\alpha} - 2)!} = \frac{\prod_{1}^{C_{\alpha}}{i}}{\prod_{1}^{C_{\alpha} - 2}{j}} = \frac{\prod_{1}^{C_{\alpha} - 2}{i} \prod_{C_{\alpha} - 1}^{C_{\alpha}}{i}}{\prod_{1}^{C_{\alpha} - 2}{j}} = \frac{\prod_{1}^{C_{\alpha} - 2}{i}}{\prod_{1}^{C_{\alpha} - 2}{j}} \prod_{C_{\alpha} - 1}^{C_{\alpha}}{i} = 1 * \prod_{C_{\alpha} - 1}^{C_{\alpha}}{i} = (C_{\alpha}) * (C_{\alpha} - 1) = C_{\alpha}^2 - C_{\alpha}$. 

* So $P_{\alpha} = {C_{\alpha} \choose 2} = \frac{C_{\alpha}!}{2! (C_{\alpha} - 2)!} = \frac{1}{2!} * \frac{C_{\alpha}!}{(C_{\alpha} - 2)!} = \frac{1}{2} * (C_{\alpha}^2 - C_{\alpha}) = \frac{C_{\alpha}^2 - C_{\alpha}}{2}$.

* Thus, $P = \sum_{\alpha \in MS}{P_{\alpha}} = \sum_{\alpha \in MS}{\frac{C_{\alpha}^2 - C_{\alpha}}{2}} = \frac{1}{2} \sum_{\alpha \in MS}{C_{\alpha}^2 - C_{\alpha}}$.

So finally:

$P = \frac{\sum_{\alpha \in MS}{C_{\alpha}^2 - C_{\alpha}}}{2}$


```python
for metasplit in metasplits:
  metasplits[metasplit].num_problems = 1/2 * sum([chars_per_alphabet[alph]**2 - chars_per_alphabet[alph] for alph in metasplits[metasplit].alphabets])
```

Also recall that problems act as samples at the Meta-level, so we may count the number of metabatches with thios information if we know the metabatch size.


```python
metabatch_size = 8
num_metabatches = int(metasplits['metatrain'].num_problems / metabatch_size)
```


```python
num_metabatches
```




    1495



### The Meta-level DataLoader

As we need to define DataLoaders when training a problem, which will generate the batches (samples + labels at each batch), at the Meta-level we will need an object that generates the meta-batches (problems at each meta-batch). We will call this the *MetaLoader*.

This Meta-Loader should give the tools to generate a DataLoader for each of its problems.

So at this point let's explore the needs to create a problem DataLoader in our context. First, we need to define which will be the alphabet we will be working on. For simplicity, we will use the Latin alphabet as toy example. We will take all the characters of this alphabet from the list.


```python
toy_alphabet = 'Latin'
toy_characters = [char for char in characters if char.split('/')[0] == toy_alphabet]
```

Now let's randomly pick 2 characters within the list.


```python
toy_problem_characters = random.sample(toy_characters, 2)
```


```python
toy_problem_characters
```




    ['Latin/character10', 'Latin/character11']



We picked both the 10th and the 11th characters from the latin alphabets. These are *J* and *K* (count it if you wish to). Let's continue to see if this matches.

Now we want to get which samples in the dataset correspond to this character. Recall that in torchvision's Omniglot, samples are sequential with 20 samples per character, so we aim to get the position of the characters in the list of characters and therefore the position of its samples in the dataset. This last value corresponds to the range of values between the character position (in the whole list) multiplied by 20 and the character position +1 multiplied by 20 (the beginning of the following character range). We want a flattened list of these sample indices for both characters in the problem.


```python
toy_problem_char_idx = [characters.index(tchar) for tchar in toy_problem_characters]  #  position of the characters in the list of characters
toy_problem_samples_idx = [toy_sample_range for tcharidx in toy_problem_char_idx for toy_sample_range in range(tcharidx * 20, (tcharidx + 1) * 20)]
```


```python
toy_problem_samples_idx
```




    [13640,
     13641,
     13642,
     13643,
     13644,
     13645,
     13646,
     13647,
     13648,
     13649,
     13650,
     13651,
     13652,
     13653,
     13654,
     13655,
     13656,
     13657,
     13658,
     13659,
     13660,
     13661,
     13662,
     13663,
     13664,
     13665,
     13666,
     13667,
     13668,
     13669,
     13670,
     13671,
     13672,
     13673,
     13674,
     13675,
     13676,
     13677,
     13678,
     13679]




```python
len(toy_problem_samples_idx)
```




    40



This length makes sense since we have 2 characters with 20 samples each.

Now we need to explicitly tell the dataloader to take among these samples. The problem is that the Omniglot dataset from torchvision cannot be split as far as we know. Instead, we may make use of the torch's [SubsetRandomSampler object](https://pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler), which explicitly tells which samples to randomly take. Furthermore, we want these samples to be taken by batches, so we will also wrap it into a [BatchSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler).


```python
indexer = BatchSampler(SubsetRandomSampler(toy_problem_samples_idx), batch_size=8, drop_last=True)
```

So now we may directly create a DataLoader with the raw Omniglot dataset using this sampler object that will directly serve us random batches of samples of the characters of our problems, as we want. We will use *shuffle=False* since the sampler already shuffles the samples.


```python
problem_loader = DataLoader(dataset=omniglot_raw, shuffle=False, batch_sampler=indexer)
```

Now let's review both samples and labels from the Data Loader. We want to verify that samples are what we expect (shape of *batch_size x num_channels x width x height*, i.e. 8 x 1 x 105 x 105), that images correspond to *J*'s and *K*'s and that labels are randomly shuffled.


```python
for ibatch, batch in enumerate(problem_loader):
  print('batch ' + str(ibatch))
  print(len(batch))
  print(batch[0].shape)
  print(batch[1])
```

    batch 0
    2
    torch.Size([8, 1, 105, 105])
    tensor([682, 683, 683, 682, 682, 683, 682, 682])
    batch 1
    2
    torch.Size([8, 1, 105, 105])
    tensor([683, 682, 683, 683, 682, 683, 683, 683])
    batch 2
    2
    torch.Size([8, 1, 105, 105])
    tensor([682, 682, 683, 682, 682, 682, 683, 683])
    batch 3
    2
    torch.Size([8, 1, 105, 105])
    tensor([683, 683, 683, 682, 683, 683, 682, 682])
    batch 4
    2
    torch.Size([8, 1, 105, 105])
    tensor([682, 683, 683, 682, 683, 682, 682, 682])



```python
plt.figure(figsize=(15,15))
columns = 8
for ibatch, batch in enumerate(problem_loader):
  for isample, sample in enumerate(batch[0]):
    plt.subplot(5, 8, (ibatch * 8) + isample + 1)
    plt.imshow(sample[0].numpy())
```


    
![png](Metabloggism_Meta-Learning_files/Metabloggism_Meta-Learning_94_0.png)
    


So as we see, everythin matches our needs.

Now that we defined a DataLoader for the problem, we are ready to create the Meta-Loaders. 

A Meta-Loader object should, at each step, be able to return a batch of problem DataLoaders. As we saw before, these DataLoaders should return at least the indices of the samples to use at each batch, and then at training/validation/test time the indices will point to the data to load from the Omniglot raw dataset. We could also work with DataLoaders that directly deliver the data, but since we would be creating lots of dataloaders with raw data, the memory cost could be too high. That is the main reason why we will work with indices instead.

So, what should the MetaLoader class contain? Making it simple, we will just need its initialization and an __iter__ method that will be returning us the problem loaders at each meta-batch. The rest should be internal methods. And how should this mandatory methods work? 

Well, when initializing a MetaLoader it will load the necessary info at both Meta-level and Learning level, as well as it will initialize a sampler as we saw before at the DataLoader but now for the Meta-Level, which will be run at *\__iter__*. This sampler will just sample from the problem indices (so one of the variables at the Meta-level must be the number of problems in the meta-set, that can be computed as explained before) for the same reason as before, memory optimization.

Then at *\__iter__* time the sampler will return a batch of problem indices. From each of these indices we will make a method (*\__get_problem_loader__* method) to get the corresponding problem loader, which will begin searching the alphabet and the characters in the alphabet for each problem index and then the samples index in this problem(*\__problem_idx_to_samples_idx__* method), for further problem loader building (*\__build_problem_loader_from_samples__* method, which will return the three loaders of the problem for train, validation and test).

Thus, when iterating the MetaLoader object, we will at each meta-batch receive a list of (meta-batch size) dictionaries with keys train, validation and test where for each key the corresponding set DataLoader of the problem will be contained.


```python
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
```

We will reimplement the *\__chars_per_alphabet__* variable by splitting it between the 3 Meta-sets.


```python
chars_per_alphabet = {split: {alph: [char.split('/')[0] for char in characters].count(alph) for alph in metasplits[split].alphabets} for split in metasplits}
```

And we will finally initialize a MetaLoader object. We will emulate a Metatrain loader.


```python
metaloader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=metabatch_size, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metatrain'], problem_ratios = [0.75, 0.15, 0.1])
```

Now let's explore the content of a Meta-batch.


```python
metabatch = next(iter(metaloader))
```


```python
metabatch[0]
```




    {'train': <torch.utils.data.dataloader.DataLoader at 0x7f1fc92d3fa0>,
     'val': <torch.utils.data.dataloader.DataLoader at 0x7f1fc92d3850>,
     'test': <torch.utils.data.dataloader.DataLoader at 0x7f1fc92d3a90>}



As expected, we successfully got a series of dataloaders for each set at each problem. Now let's check the content of the samples in the problems of the metabatch. This means that we want to plot all the (shuffled) samples of each problem, so we should plot a collection of 8 grids where each grids contains a list of images among 2 samples. 


```python
plt.figure(figsize=(15,100))
columns = 8
for imb in range(8):
    for ibatch, batch in enumerate(metabatch[imb]['train']):
        for isample, sample in enumerate(batch[0]):
            plt.subplot(40, 8, (imb * 40) + (ibatch * 8) + isample + 1)
            plt.imshow(sample[0].numpy())
```


    
![png](Metabloggism_Meta-Learning_files/Metabloggism_Meta-Learning_107_0.png)
    


So we again got what we expected. Thus, we will finally define our MetaLoaders.


```python
metatrain_loader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=metabatch_size, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metatrain'], problem_ratios = [0.75, 0.15, 0.1])
metaval_loader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=metabatch_size, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metaval'], problem_ratios = [0.75, 0.15, 0.1])
metatest_loader = MetaLoader(base_dataset=omniglot_raw, metabatch_size=1, batch_sizes={'train': 8, 'val': 1, 'test': 1}, chars_per_alphabet=chars_per_alphabet['metatest'], problem_ratios = [0.75, 0.15, 0.1])
```

### Training a model in a problem

So at this point we are ready to use a single problem extracted from the train MetaLoader and train a model on it. First, we should get our DataLoaders


```python
toy_metabatch = next(iter(metatrain_loader))
```


```python
toy_problem_loader = toy_metabatch[0]['train']
toy_problem_loader_val = toy_metabatch[0]['val']
toy_problem_loader_test = toy_metabatch[0]['test']
```


```python
for i, data in enumerate(toy_problem_loader, 0):
    for isample, sample in enumerate(data[0]):
        plt.subplot(3, 8, (i * 8) + isample + 1)
        plt.imshow(sample[0].numpy())
```


    
![png](Metabloggism_Meta-Learning_files/Metabloggism_Meta-Learning_114_0.png)
    


Now we will define a Simple Neural network (it should be enough for the simple 105x105 character classification problems).


```python
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
```

Now we will define also our Loss function as well as our optimizer (we will use SGD for simpllicity). We will train 15 epochs in the problem and run a validation step at the end of an epoch.


```python
model = SimpleNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```


```python
n_epochs = 15
```

We will need the following methods to process both labels and samples (for the labels we just one to convert both labels in 0 and 1 respectively).


```python
def process_labels(labels_raw, ref_label):
  return (labels_raw == ref_label).float()

def preprocess_inputs(inputs):
    return (1- inputs) * 255
```

So we are ready to implement and run what we defined.


```python
ref_label = None
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}')
    running_loss = 0.0
    val_loss = 0.0
    val_accuracy = 0.0
    for i, data in enumerate(toy_problem_loader, 0):
        inputs_raw, labels_raw = data
        optimizer.zero_grad()
        inputs = preprocess_inputs(inputs_raw)
        outputs = model(inputs)
        if ref_label is None:
            ref_label = labels_raw[0]
        labels = process_labels(labels_raw, ref_label)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        accuracy = (((1 - outputs) < outputs).float() == labels).sum() / outputs.shape[0]
        print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
    for iv, datav in enumerate(toy_problem_loader_val):
        inputs_rawv, labels_rawv = datav
        inputsv = preprocess_inputs(inputs_rawv)
        outputsv = model(inputsv)
        labelsv = process_labels(labels_rawv, ref_label)
        lossv = criterion(outputsv, labelsv[0])
        val_loss += lossv.item()
        val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()
    print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')

print('Finished Training')
```

    Epoch 1


    /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1967: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")


    Epoch 1, step     1], Loss: 0.6658390164375305, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6492561101913452, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.8146457672119141, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6260940631230673, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.60048508644104, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.5681989789009094, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6169461607933044, Accuracy: 0.875
    Epoch 2, VALIDATION], Loss: 0.5620179573694865, Accuracy: 1.0
    Epoch 3
    Epoch 3, step     1], Loss: 0.6216239333152771, Accuracy: 1.0
    Epoch 3, step     2], Loss: 0.5817804336547852, Accuracy: 1.0
    Epoch 3, step     3], Loss: 0.507636308670044, Accuracy: 1.0
    Epoch 3, VALIDATION], Loss: 0.5611441731452942, Accuracy: 0.8333333134651184
    Epoch 4
    Epoch 4, step     1], Loss: 0.5516251921653748, Accuracy: 1.0
    Epoch 4, step     2], Loss: 0.5512588024139404, Accuracy: 1.0
    Epoch 4, step     3], Loss: 0.4609960913658142, Accuracy: 1.0
    Epoch 4, VALIDATION], Loss: 0.593102385600408, Accuracy: 0.8333333134651184
    Epoch 5
    Epoch 5, step     1], Loss: 0.4159203767776489, Accuracy: 1.0
    Epoch 5, step     2], Loss: 0.6008561849594116, Accuracy: 1.0
    Epoch 5, step     3], Loss: 0.5070333480834961, Accuracy: 1.0
    Epoch 5, VALIDATION], Loss: 0.5653840551773707, Accuracy: 0.8333333134651184
    Epoch 6
    Epoch 6, step     1], Loss: 0.5032209753990173, Accuracy: 1.0
    Epoch 6, step     2], Loss: 0.5032068490982056, Accuracy: 1.0
    Epoch 6, step     3], Loss: 0.4557313323020935, Accuracy: 1.0
    Epoch 6, VALIDATION], Loss: 0.6277975539366404, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.645679771900177, Accuracy: 1.0
    Epoch 7, step     2], Loss: 0.45026305317878723, Accuracy: 0.875
    Epoch 7, step     3], Loss: 0.5558935403823853, Accuracy: 0.875
    Epoch 7, VALIDATION], Loss: 0.6248213152090708, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.4557203948497772, Accuracy: 1.0
    Epoch 8, step     2], Loss: 0.5506904125213623, Accuracy: 1.0
    Epoch 8, step     3], Loss: 0.45581647753715515, Accuracy: 1.0
    Epoch 8, VALIDATION], Loss: 0.562673901518186, Accuracy: 0.8333333134651184
    Epoch 9
    Epoch 9, step     1], Loss: 0.5032058358192444, Accuracy: 1.0
    Epoch 9, step     2], Loss: 0.5032044649124146, Accuracy: 1.0
    Epoch 9, step     3], Loss: 0.5506901144981384, Accuracy: 1.0
    Epoch 9, VALIDATION], Loss: 0.5044418474038442, Accuracy: 1.0
    Epoch 10
    Epoch 10, step     1], Loss: 0.4557187557220459, Accuracy: 1.0
    Epoch 10, step     2], Loss: 0.5032044649124146, Accuracy: 1.0
    Epoch 10, step     3], Loss: 0.5981758236885071, Accuracy: 1.0
    Epoch 10, VALIDATION], Loss: 0.5032213479280472, Accuracy: 1.0
    Epoch 11
    Epoch 11, step     1], Loss: 0.4557187557220459, Accuracy: 1.0
    Epoch 11, step     2], Loss: 0.5032044649124146, Accuracy: 1.0
    Epoch 11, step     3], Loss: 0.5506902933120728, Accuracy: 1.0
    Epoch 11, VALIDATION], Loss: 0.5032058407862982, Accuracy: 1.0
    Epoch 12
    Epoch 12, step     1], Loss: 0.5506907105445862, Accuracy: 1.0
    Epoch 12, step     2], Loss: 0.5032053589820862, Accuracy: 1.0
    Epoch 12, step     3], Loss: 0.5032044053077698, Accuracy: 1.0
    Epoch 12, VALIDATION], Loss: 0.5032146573066711, Accuracy: 1.0
    Epoch 13
    Epoch 13, step     1], Loss: 0.5506961941719055, Accuracy: 1.0
    Epoch 13, step     2], Loss: 0.5032044649124146, Accuracy: 1.0
    Epoch 13, step     3], Loss: 0.5507189035415649, Accuracy: 1.0
    Epoch 13, VALIDATION], Loss: 0.5032911549011866, Accuracy: 1.0
    Epoch 14
    Epoch 14, step     1], Loss: 0.5508079528808594, Accuracy: 1.0
    Epoch 14, step     2], Loss: 0.5032503604888916, Accuracy: 1.0
    Epoch 14, step     3], Loss: 0.455719918012619, Accuracy: 1.0
    Epoch 14, VALIDATION], Loss: 0.5034261147181193, Accuracy: 1.0
    Epoch 15
    Epoch 15, step     1], Loss: 0.5033642053604126, Accuracy: 1.0
    Epoch 15, step     2], Loss: 0.4083341956138611, Accuracy: 1.0
    Epoch 15, step     3], Loss: 0.550751805305481, Accuracy: 1.0
    Epoch 15, VALIDATION], Loss: 0.5033487677574158, Accuracy: 1.0
    Finished Training


As we can see, the model easily learns and in y case it reached an almost perfect accuracy. But things will get messy.

### Train with manual GD algorithm

Before concluding I am a maniac by trying this for no reason, let's take a look at [MAML official implementation on Github](https://github.com/cbfinn/maml). In the [*maml.py*](https://github.com/cbfinn/maml/blob/master/maml.py#L79) module you can realize that at Learning level, authors also update manually the weights at each step. The same goes for [unofficial Pytorch implementations](https://github.com/dragen1860/MAML-Pytorch). Why is this? Well, if you think, in Meta-Learning, the model must learn at 2 levels. I.e. gradients, losses, optimizers, etc must work at 2 levels at the same time, which ML frameworks are not made for. Updating the weights manually in one of both levels is actually a workaround to this issue, so from Pytorch perspective there is no Learning level, and it just knows the updates at Meta-Learning level.

So, before going for the Meta-Learning level we will manually train the same as before but now with manual updates. Note that updating model's parameters is done though the code chunk:

```
for param, param_key in zip(new_weights, param_keys):
            model._modules[param_key[0]]._parameters[param_key[1]] = param

```


```python
model = SimpleNet()
criterion = nn.BCEWithLogitsLoss()
update_lr = 0.01

ref_label = None
param_keys = [(mod, kname) for mod in model._modules for kname in model._modules[mod]._parameters]
new_weights = model.parameters()
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}')
    running_loss = 0.0
    val_loss = 0.0
    val_accuracy = 0.0
    for i, data in enumerate(toy_problem_loader, 0):
        inputs_raw, labels_raw = data
        inputs = preprocess_inputs(inputs_raw)
        outputs = model(inputs)
        if ref_label is None:
            ref_label = labels_raw[0]
        labels = process_labels(labels_raw, ref_label)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, model.parameters())
        new_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grads, new_weights)))
        running_loss += loss.item()
        accuracy = (((1 - outputs) < outputs).float() == labels).sum() / outputs.shape[0]
        for param, param_key in zip(new_weights, param_keys):
            model._modules[param_key[0]]._parameters[param_key[1]] = param
        print(f'Epoch {epoch + 1}, step {i + 1:5d}], Loss: {loss.item()}, Accuracy: {accuracy}')
    for iv, datav in enumerate(toy_problem_loader_val):
        inputs_rawv, labels_rawv = datav
        inputsv = preprocess_inputs(inputs_rawv)
        outputsv = model(inputsv)
        labelsv = process_labels(labels_rawv, ref_label)
        lossv = criterion(outputsv, labelsv[0])
        val_loss += lossv.item()
        val_accuracy += (((1 - outputsv) < outputsv).float() == labelsv).sum()
    print(f'Epoch {epoch + 1}, VALIDATION], Loss: {val_loss / (iv + 1)}, Accuracy: {val_accuracy / (iv + 1)}')

print('Finished Training')
```

So training is also completed correctly this way.

### Meta-Learning training

At this point we will directly define the Meta-Learning pipeline as follows:

*   At each Meta-epoch, the Meta-train MetaLoader is called at each Meta-train meta-step.
*   At each Meta-train meta-step, a each problem is the meta-batch is asked for its DataLoaders
*   For the train DataLoader, at each epoch each step is called and the batch is predicted.
*   Manually, at this step the model weights are updated
*   In the end of the epoch, a validation is run
*   In the last epoch, the validation loss is computed normally but the model is returned to its original (at the beginning of the problem) state.
*   The same is repeated over all the problems in the meta-batch and all final validation losses are averaged.
* A pyorch update is performed in the end of the meta-batch
* Every 1000 meta-steps a Meta-Validation meta-step is performed. The process is the same as in Meta-train but the loss won't update the model (only computed for Meta-training guidance matters).



```python
def make_step(model, outputs, labels, update_lr, in_weights):
    loss = criterion(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters())
    out_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grads, in_weights)))
    accuracy = (((1 - outputs) < outputs).float() == labels).sum() / outputs.shape[0]
    return out_weights, loss, accuracy
```


```python
def update_model(model, new_weights, param_keys):
    for param, param_key in zip(new_weights, param_keys):
        model._modules[param_key[0]]._parameters[param_key[1]] = param
```


```python
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

    ===============================
    //           Meta-Epoch 1       //
    ===============================
    0 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.7687771320343018, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6925913095474243, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6927804350852966, Accuracy: 0.5


    /usr/local/lib/python3.8/dist-packages/torch/nn/functional.py:1967: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")


    [1;30;43mStreaming output truncated to the last 5000 lines.[0m
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    1036 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.8333333134651184
    1037 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6932497620582581, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931473016738892, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 4
    Epoch 4, step     1], Loss: 0.6932467222213745, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931473016738892, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931473016738892, Accuracy: 0.75
    Epoch 5, step     2], Loss: 0.6932439208030701, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6932413578033447, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931473016738892, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6932388544082642, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931473016738892, Accuracy: 0.25
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     3], Loss: 0.6932364702224731, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 9
    Epoch 9, step     1], Loss: 0.6932342648506165, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931473016738892, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6932321190834045, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6932300925254822, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.6931473016738892, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931473016738892, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931473016738892, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6932281851768494, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 14
    Epoch 14, step     1], Loss: 0.6932263374328613, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931473016738892, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     3], Loss: 0.6932245492935181, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 1.0
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.693159818649292, Accuracy: 0.875
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.693159818649292, Accuracy: 0.75
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.693159818649292, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.693159818649292, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.693159818649292, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.693159818649292, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.693159818649292, Accuracy: 0.75
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.693159818649292, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.693159818649292, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471228599548, Accuracy: 0.75
    Epoch 1, step     3], Loss: 0.6931470036506653, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931469440460205, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471228599548, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931469440460205, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471228599548, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931470036506653, Accuracy: 0.75
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471228599548, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 6, step     3], Loss: 0.6931470036506653, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931469440460205, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471228599548, Accuracy: 0.75
    Epoch 8, step     3], Loss: 0.6931470036506653, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931469440460205, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471228599548, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931470036506653, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471228599548, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931470036506653, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471228599548, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931470036506653, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471228599548, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931470036506653, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471228599548, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931470036506653, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     3], Loss: 0.6931471228599548, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931472420692444, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931472420692444, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    1038 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    1039 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931465268135071, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471228599548, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6929552555084229, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471228599548, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.693013608455658, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931463479995728, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6929985284805298, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6930403709411621, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931462287902832, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6930311918258667, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6929515600204468, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931459903717041, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6929175853729248, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6929798126220703, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6928402781486511, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931451559066772, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931452751159668, Accuracy: 0.75
    Epoch 8, step     3], Loss: 0.6929152011871338, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931449174880981, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.692875862121582, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6928192377090454, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6924663782119751, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     2], Loss: 0.6925125122070312, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931395530700684, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6914534568786621, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     3], Loss: 0.6931471228599548, Accuracy: 0.25
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 13, step     2], Loss: 0.6931257247924805, Accuracy: 0.25
    Epoch 13, step     3], Loss: 0.6903964877128601, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6930469870567322, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6734169125556946, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6456539630889893, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931468447049459, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6456685662269592, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931453347206116, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6456435322761536, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931467950344086, Accuracy: 0.3333333432674408
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931618452072144, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.7013236880302429, Accuracy: 0.75
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.7013384103775024, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.7013236880302429, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931618452072144, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931618452072144, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931618452072144, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.7013236880302429, Accuracy: 0.75
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.7013236880302429, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931618452072144, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.7013236880302429, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6931618452072144, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     3], Loss: 0.7013384103775024, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931618452072144, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.7013236880302429, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931618452072144, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931618452072144, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.7013236880302429, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931618452072144, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.7013236880302429, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.7013383507728577, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931618452072144, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.7013384103775024, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    1040 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931450963020325, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931450963020325, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931450963020325, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931450963020325, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931450963020325, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931450963020325, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931450963020325, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931450963020325, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931450963020325, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931450963020325, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    1041 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931325197219849, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931325197219849, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.685472846031189, Accuracy: 0.125
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931325197219849, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.685472846031189, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.685472846031189, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931325197219849, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.685472846031189, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931324601173401, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931325197219849, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.685472846031189, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6854581832885742, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6854581832885742, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931325197219849, Accuracy: 0.75
    Epoch 10, step     2], Loss: 0.685472846031189, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931325197219849, Accuracy: 0.25
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.685472846031189, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.685472846031189, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.685472846031189, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931325197219849, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6854581832885742, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.685472846031189, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931325197219849, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    1042 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931456327438354, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.5982619524002075, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931469440460205, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6456738114356995, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931468844413757, Accuracy: 0.75
    Epoch 2, step     3], Loss: 0.6457263231277466, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931469440460205, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     2], Loss: 0.693146824836731, Accuracy: 0.25
    Epoch 3, step     3], Loss: 0.6457088589668274, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931469241778055, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6457077860832214, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6456754803657532, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931454539299011, Accuracy: 0.75
    Epoch 4, VALIDATION], Loss: 0.6931469142436981, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931309103965759, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.5982460975646973, Accuracy: 0.875
    Epoch 5, step     3], Loss: 0.6931453943252563, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931469043095907, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931443214416504, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6456587910652161, Accuracy: 0.625
    Epoch 6, step     3], Loss: 0.6457151770591736, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931468844413757, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6456596255302429, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931452751159668, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6457098126411438, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931468745072683, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.645708441734314, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6456731557846069, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.693132221698761, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931468645731608, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931296586990356, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.5982348322868347, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931468447049459, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931437253952026, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6456550359725952, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931468447049459, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, step     2], Loss: 0.645706832408905, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6456537246704102, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931468347708384, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6457046270370483, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6456717252731323, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931255459785461, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.693146824836731, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931438446044922, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931449770927429, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6457033753395081, Accuracy: 0.75
    Epoch 13, VALIDATION], Loss: 0.693146804968516, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6456713080406189, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931432485580444, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.693146804968516, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6456672549247742, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931449174880981, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6457016468048096, Accuracy: 0.875
    Epoch 15, VALIDATION], Loss: 0.6931467950344086, Accuracy: 0.6666666865348816
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.5
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 7 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6485647559165955, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6485647559165955, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6485647559165955, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     3], Loss: 0.6485647559165955, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     2], Loss: 0.6485647559165955, Accuracy: 0.75
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     3], Loss: 0.6485647559165955, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6485647559165955, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.6485647559165955, Accuracy: 0.75
    Epoch 9, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6485647559165955, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6485647559165955, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6485647559165955, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6485647559165955, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471725304922, Accuracy: 0.6666666865348816
    - Problem 8 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.1666666716337204
    1043 updates at Meta-Level
    - Problem 1 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 2 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 3 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.875
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    - Problem 4 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 1, step     2], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 1, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 2, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 3, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 4, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 5, step     3], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 5, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 6, step     3], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 6, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 7, step     2], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 7, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 8, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 9, step     2], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 9, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 10, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 11, step     3], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 11, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 12, step     2], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 12, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 13, step     3], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 13, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 14, step     3], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 14, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 15, step     2], Loss: 0.6931362152099609, Accuracy: 0.0
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.0
    Epoch 15, VALIDATION], Loss: 0.6931325793266296, Accuracy: 0.0
    - Problem 5 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.125
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 14, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    Epoch 15
    Epoch 15, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 15, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 15, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 15, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.6666666865348816
    - Problem 6 -
    Epoch 1
    Epoch 1, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 1, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 1, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 2
    Epoch 2, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 2, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 2, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 2, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 3
    Epoch 3, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 3, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 3, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 4
    Epoch 4, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 4, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 4, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 5
    Epoch 5, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 5, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 5, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 6
    Epoch 6, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 6, step     2], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 6, step     3], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 6, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 7
    Epoch 7, step     1], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 7, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 8
    Epoch 8, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 8, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 8, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 8, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 9
    Epoch 9, step     1], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 9, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 9, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 9, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 10
    Epoch 10, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 10, step     2], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 10, step     3], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 10, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 11
    Epoch 11, step     1], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 11, step     2], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, step     3], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 11, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 12
    Epoch 12, step     1], Loss: 0.6931471824645996, Accuracy: 0.75
    Epoch 12, step     2], Loss: 0.6931471824645996, Accuracy: 0.375
    Epoch 12, step     3], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 12, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 13
    Epoch 13, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 13, step     2], Loss: 0.6931471824645996, Accuracy: 0.5
    Epoch 13, step     3], Loss: 0.6931471824645996, Accuracy: 0.25
    Epoch 13, VALIDATION], Loss: 0.6931471824645996, Accuracy: 0.3333333432674408
    Epoch 14
    Epoch 14, step     1], Loss: 0.6931471824645996, Accuracy: 0.625
    Epoch 14, step     2], Loss: 0.6931471824645996, Accuracy: 0.5


As said before, we don't want to check the performance of the approach since we will need deep review and debug of the behaviour (with proper hyperparameter search, etc) and this is out of the scope of this post. However, I am glad to see that I have a running and working Meta-Learning pipeline.

In the next post, my intention is to see make a performance analysis of this and compare it to other approaches.

Again, thank you for your attention and we'll meet in the next post!
