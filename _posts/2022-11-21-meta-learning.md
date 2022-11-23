# **Meta-Learning explained**

## **What will be reviewed**

In this first post (aside from the Welcome one) I will expose a brief summary of Meta-Learning. It was one of the topics I researched for [my Master’s dissertation](https://upcommons.upc.edu/bitstream/handle/2117/179428/cMas.pdf;jsessionid=18807FB3EE2D5343E5F0DF9A5BA37D7F?sequence=1) (and one of the main topics I actually developed it about) and since then, one of the topics I am more interested in.

As you will see, the summary will not be the latest trend, because I want to give it a more historical explanation. The State of the Art may be reviewed in a different post. Instead, this post will focus on taking an interesting tour along the Meta-Learning evolution and understanding the context to build more efficiently in the future.

## **What is Meta-Learning?**

Any task or process can involve a process of learning in order to improve performance in it. Systems are no strangers to this, for example, imagine a system that has to perform some kind of face identification. The system will perform better when it has lived a learning process before. Well, that is exactly the point of Machine Learning, isn't it? But let's switch the focus. Learning itself is a process. So according to the previous statement, it can also be learned. And this is the exact definition of Meta-Learning: Learning to Learn.

![](https://i.imgur.com/Wc5zMl2.png)

| <b>Meme, credits to [@joavanschoren](https://twitter.com/joavanschoren)... but I cut out the end since it contained a spoiler of content in the post below</b> |
  
From another perspective, we humans do not learn most things from scratch. For example, if I present you with a new bird species (unknown to you) and tell you "Have you seen this bird?" you will probably learn its patterns with just a quick look. This is because you have seen so many birds in your life, and you'll directly look at the differential features (feathers color, feet, beak shape...) to absorb the information. Also, you may understand the context (e.g. identify how does it fly with respect to the ground, wind, etc) because you already know things about this context. In contrast, if you present it to a baby, he will not understand at all because he still has to learn everything. Another simple example is how before analyzing a book from a literature perspective a kid has to learn how to read.

Thus, Meta-Learning is extending the Learning process to one level above, and Learning to Learn. And, how does that contribute? To make it simple, it makes the Learning process more efficient (in any way, which could be faster, more stable, more qualitative...), and this allows us to overcome some important issues (we'll discuss that later). When looking at that, we could say we work at 2 levels, the Learning level and the Meta-Learning level. You may also note that this can be even extended one level above since Learning to Learn is another process. You are right, this can be done and we then would achieve the Meta-meta-Learning level, thus working at 3 levels. And this is also extendible to any level until infinity (so yes, you could learn how to learn how to learn how to learn...how to learn). However, such a process is obviously limited by our capacity. We as humans, apply this at many levels, but it is also limited by our brain capacity. From a system perspective, it is limited by its computational power.

So that said, how does this fit in out Machine Learning (ML) interest? Well, there are different ways of applying Meta-Learning in ML that will be reviewed in this post, but keep in mind a setting where an inner algorithm works for a prediction task. This algorithm learns under some conditions, by updating some model by some Learning Rule depending in some data. However, in a vanilla ML setting this conditions are usually picked manually and suboptimal. In Meta-learning there would also be an outer optimizer whose task is to optimize these conditions. Some ways to achieve it could be updating the Learning Rule, selecting the model (architecture or initial parameters) or rescheduling the data.
  
![Imgur](https://i.imgur.com/zyRBmGS.png)

Until this point, there could be a bit of confusion between Meta-Learning and other techniques. The difference is how is the schedule built and where is the data taken from. In Meta-Learning the flow works as follows:

> Pretend we are aiming to solve an specific task. This task may be drawn from a bigger domain of tasks. In the example below, imagine we face a binary image classification task among a series of animal classes (monkey, dog, cat, elephant, fish, snake, hypo...). For example, imagine that we in the end will end up having to classify between dogs and snakes. We want to learn how to learn efficiently this specific task. We could define the domain as binary animal image classification tasks. The domain also includes a series of conditions below (RGB camera images, full body, denoised, real...). Now, along all the domain we may draw a series of tasks different than the one we are aiming to solve. To avoid this happening we may drop both dog and snake classes from a bag with all classes, and build tasks by picking combinations of two classes. Thus, for each task, we will have to classify images between both classes and then that will be a binary animal image classification task. These tasks will be equivalent to the samples in the Learning level (or task level) but in the Meta-Learning level. So, equivalently to what we would do at the training level we will build two (meta-)sets. One will have Meta-training tasks while the other Meta-test tasks. And yeah, if you do things correctly you would also have a Meta-validation meta-set, of course. Then, for each task we will work as always at the Learning Level, getting samples for the task and splitting them between train, validation and test (as always). At this level, we will train the model as usual, evaluate, etc., so we will end up having some performance measure (usually a Loss value). These individual task results will serve in the Meta-Learning level to evaluate the outer optimizer and making the corresponding updates. Below in the example, each task updates the model, but it is just an example. Actually, just like at the Learning level, it can work by batches (in this case, batches of tasks). Just good luck with your hardware limitations. Then the test set is used to evaluate by any given metric.
  
![Imgur](https://i.imgur.com/a9Fr97l.png)

Note that all this process is designed from the beginning to optimize the process of Learning in our target task. So our Meta-Learning schedule is indeed a schedule for Learning to Learn. Any approach that falls into that definition is a Meta-Learning approach. You must also notice the difference between that and other similar techniques. For example, in Transfer Learning (another different whole topic) you do not learn how to learn for an specific task (or a task from an specific domain), but instead use old knowledge to get closer to the optimal solution (when you learned that old knowledge, you learned it for a whole different solution and was not intended to extend to any other different problem, thus there doesn't exist a Meta-Learning level). Some related topics are:

* Transfer Learning: using old knowledge optimized to solve another task to solve the current target task.
  
![Imgur](https://i.imgur.com/4kN4Xu8.png)
  
* Domain adaptation: learning from a different domain with some common conditions (e.g. in our example above learning the dog vs snake task from synthetic images).
  
![Imgur](https://i.imgur.com/FLTKEbi.png)
  
* Multimodal Learning: learning from different modes of data (e.g. a text description), although it can be viewed from a Meta-Learning level it covers a different topic and is treated differently (since it builds a different setting).
  
![Imgur](https://i.imgur.com/rTmjzmU.png)

Ok, so now we have an idea of what is Meta-Learning and how to use it. But why use it? Motivations are diverse and have varied over time. Actually, one can use it whenever it is beneficial for his task. But the important question is what did raise the interest of researchers to present schedules, definitions and solutions that include Meta-Learning? To do so, we may have a quick recap of Meta-Learning history.

## **Origins of Meta-Learning**

The term arose in a publication in Jürgen Schmidhuber's thesis called [*Evolutionary Principles in Self-Referential Learning*](https://people.idsia.ch/~juergen/diploma1987ocr.pdf) (1987). The paper is incredibly dense, but a mine of knowledge and talks about some deep topics such as Information, Entropy, Evolution... We may talk specifically about this paper in future posts, but what concerns us now is that it presented the idea of Meta-Learning as a way to modify the plans (the equivalent of what the schedulers + optimizers in ML mean nowadays) in order to generalize to a whole group of domains. This group is again some kind of (meta-)domain, so it also needed some (meta-)plan. The hypothesis that Schmidhuber did back in 1987 is that this concept was extendable to infinite levels of abstraction, thus allowing the definition (apart from the domain level and the meta-level) of a meta-meta-level, a meta-meta-meta-level, and so on, although the paper also points the compromise that the realization has with the computation capacity of the hardware (plus recall we are in 1987, there were no NVIDIA RTX 4090... actually [the first GPU](https://en.wikipedia.org/wiki/GeForce_256) came out 12 years later). He proposed 2 ways to implement Meta-Learning:

* First, as a Genetic Algorithm. My interpretation is that he was wondering about what nowadays is Curriculum Learning, but from a Meta-Learning perspective. He proposed that at the Meta-Level the plan should schedule the best samples for the domain level. His own concerns? About "nature" (AI then aimed more to mimic true intelligence) and feasibility.
* Second, as a hierarchy of classifiers building the Genetic Algorithm, which will act at the meta-level. Schmidhuber pointed out that this way the mechanism could work with a fixed number of levels, just the domain level (each classifier) and the meta-level (the Genetic Algorithm).

![Imgur](https://i.imgur.com/KWobPej.png)
![Imgur](https://i.imgur.com/blCSh2S.png)

Another interesting interpretation from the paper is the way a plan works in a Genetic Algorithm, which is similar to the environment in Reinforcement Learning. Thus, surviving a plan in a Genetic Algorithm can be viewed as getting a reward in Reinforcement Learning.

5 years later, Schmidhuber made another important contribution to Meta-Learning in [*Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks*](https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf). He defined a series of sequential (by episodes or plain timesteps) problems. To make you an idea, one of these problems consisted in predicting the parking slot where some car parked given a series of sensor states (distributed along the parking ground) at different time instants. In this case, no episodes were used as it was an online problem (prediction was made at the same training time). Instead, a prediction network was used for the task. The particularity is that another network at a level above learned the weights updates that the domain one should experiment. Thus, the sequence could be seen as an artificial meta-level, while at each timestep the inner network performed the task. In the offline setting, the behavior was the same, just defining the sequence through bounded episodes. This was the first published Meta-Learning approach that worked for practical tasks. Surprisingly, [the word meta is missing](https://imgur.com/9uvwpUb) in that paper, but the interpretation seems clear to me to give the idea of an inner and an outer model that Schmidhuber was working around at that time, isn't it? When I researched about that, [Schmidhuber himself considered this as a way of Meta-Learning](https://people.idsia.ch/~juergen/metalearning.html).
  
![Imgur](https://i.imgur.com/tDtaaKC.jpg)

But that was not the only meaningful publication of Meta-Learning in 1992. Bengio (both Samy and Yoshua) et al. published this same year the paper called [*On the Optimization of a Synaptic Learning Rule*](http://www.iro.umontreal.ca/~lisa/pointeurs/bengio_1995_oban.pdf), which probably is the first definition of a Meta-Learning setting how we imagine that nowadays (although again they do not refer to the word meta!). They just defined a framework with an inner prediction algorithm (the previously called domain level), which actually was a Neural Network (yes, Neural Networks existed before LeNet, didn't you know?) and was optimized through a Synaptic Learning Rule, which again was optimized by an outer optimization algorithm (what would be the meta-level). The point of the paper is that this Synaptic Learning Rule should be parametric, so the outer optimizer should just update its parameters. Thus, by defining well the episodic nature of the updates at the beginning of the whole process, the Synaptic Learning Rule should be able to learn from the task results a generalization of the whole domain of tasks.
  
![Imgur](https://i.imgur.com/aw60MAG.png)

For sure 1992 was an important year for Meta-Learning! It was also the year I was born, so maybe it was my destiny to study this field.

Later on, Schmidhuber continued his study in Meta-Learning by extending it to Meta-Reinforcement Learning with publications such as [*On learning how to learn learning strategies*](https://people.idsia.ch/~juergen/fki198-94.pdf) (1994) or [*What's interesting*](https://people.idsia.ch/~juergen/interest.html) (1997), but we will skip this part since it falls more into the domain of Reinforcement Learning. Just recall the analogy I mentioned above between Meta-Learning in Reinforcement-Learning and Meta-Learning in Genetic Algorithms. It seems he was already pointing in that direction, and that was actually one of the main trends of Meta-Learning at that time. However, another direction was the one initiated by Bengio brothers back in 1992, and that was the one that brought us to the point we are nowadays.

In that sense, Hochreiter made another interesting publication in 2001, called [*Learning To Learn Using Gradient Descent*](https://www.researchgate.net/publication/225182080_Learning_To_Learn_Using_Gradient_Descent), where they used the aforementioned paradigm of the inner prediction algorithm and the outer optimizer (of the parametric Learning Rule) and proposed that this outer optimizer should be updated by a Gradient Descent. Oh, and finally they called that Meta-Learning. They proposed for this a task with sequences and used a Neural Network (don't be surprised, LeNet already existed) to perform the experiments. Both the inner predictor (task level) and the outer optimizer (meta-level) were RNNs. Before this publication, a similar approach was already studied called Adaptive Learning, which already used a Neural Network to optimize a learning rule. However, the setting there is different (no Meta-Learning at all).
  
![Imgur](https://i.imgur.com/8bSfzGl.png)

After that, obviously more publications about Meta-Learning appeared. However, until 2015 the focus of interest in Machine Learning was on other topics (you know that it was a time of big changes, where the first truly big Neural Networks arrived, and everything began to explode), and I don't feel that this publications repercussion on the evolution of Meta-Learning is worth enough to include in this basic summary. So with this, I think we already have an idea of how the knowledge about Meta-Learning arrived in 2015, and how it was viewed back then when the interest returned with new motivations.

## **The comeback of Meta-Learning and its relation to Few-Shot Learning**

The interest in Meta-Learning returned when ML research gazed a further step than plain basic ML tasks. Before 2015, most of the applications relied on vast amounts of data, but at the time of making ML accessible to anyone (not just the big fishes in the industry), that scenario was not realistic. Yes, there were already public datasets, but when trying to make some slightly ambiguous applications, it was needed some data conditions that were not easy to find. Not all small companies or particular researchers had access to a batch of 1 million images of, let's say, water impurities, and it was a too concrete phenomenon to find a huge open dataset about it.

In 2015, Koch et al. presented the publication [*Siamese Neural Networks for One-shot Image Recognition*](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). As the title says, they introduced the concept of One-shot Learning where they proposed to solve a task (in the paper an image classification task) with just one single sample (image) per class. They argued that humans are able to do so (e.g., in the example we made before about recognizing a new bird, we may recognize it after seeing it just one time, at least with a good memory capacity!) so why couldn't a system do so automatically? In this paper Meta-Learning was not mentioned (they just described a method to match patterns efficiently with one single image in the Omniglot dataset), but the seed of curiosity was already planted to raise interest in this topic. It is not hard to imagine that the Meta-Learning term returned to action after that. In the end, we defined before that Meta-Learning was intended to use in order to make systems able to learn tasks more efficiently, and that included several interests. One of that interests may be precisely One-shot Learning. So from the more generic old definition, now we have a concrete motivation.
  
![Imgur](https://i.imgur.com/OktLV6n.png)

We will continue in the next section reviewing how Meta-learning was proposed after that, but first I would like to discuss the feasibility of One-shot Learning. Solving One-shot Learning problems using Meta-Learning is yet another, although possible, ideal case. One will not always dispose of enough data from the task domain that will allow performing Meta-Learning efficiently, and not always a representative enough task domain may be defined. Reality may be sometimes demoralizing, and yes, expecting to approach an ideal solution with another ideal approach is not exactly a guarantee of success. But there is still a motivation behind. One-shot Learning may be a too ambitious purpose in some cases, but if the term One-shot comes from one single sample, doesn't exist a Two-shot, or a Three-shot, and so on? Yes, it does. Actually, One-shot Learning is a particularization of Few-shot Learning or K-shot Learning, which mean learning with just few or K samples respectively. Thus, the level of ambition is adjustable to the resources and needs in each case. For example, maybe we would not be able to have 1 million images of water impurities, but getting 100 labeled images of them may be more accessible, as well as a 100-shot learning problem may be more realistic to solve than a One-shot one.

So to conclude this section, Meta-Learning re-arose as a solution to Few-shot Learning. Said that I think it is important to not lose the old, more general perspective of it. That allows us to apply Meta-Learning for several problems such as Active Learning, Curriculum Learning, etc. But let's continue the story.

## **The modern Meta-Learning approaches presented**

If you search for [Meta-Learning publications since 2016](https://imgur.com/fzTtsDP), you may fall off your seat. However, what actually happened is that the modern strategies to perform Meta-Learning (most of them focused on One-Shot Learning) exploded. The main Meta-Learning strategies nowadays are divided into 3 or 4 "families". Apart from the solutions that we will review in this section, the rest of the publications focus on experimenting with them and studying the behavior, modifying them with some witty hacks, or, what is most common nowadays, trying to use them in specific scenarios.

### Usage of memories

Santoro et al. were the first (as far as I know) to refer to Meta-Learning for solving the One-shot Learning issue with [*Meta-Learning with Memory-Augmented Neural Networks*](https://proceedings.mlr.press/v48/santoro16.pdf) (2016). They propose the architecture MANN as a modification of Neural Turing Machine (Graves et al. 2014)  to achieve Meta-Learning by reducing complexity of the original mechanism, thus allowing to learn in fewer steps. Recall that one of the requirements of Deep Learning refers to the size of the dataset. There's a great post about [NTM and MANN](https://rylanschaeffer.github.io/content/research/one_shot_learning_with_memory_augmented_nn/main.html). It is a complex mechanism and we may talk a lot about this, but I'd suggest to skip this part yet (the topic is interesting but may be better to learn about this another day, you don't want to overwhelm today after reading this post) and only have in mind this first approach to achieve Meta-Learning by learning a storage mechanism. There's a discussion about if this may be considered Meta-Learning or not.
  
![Imgur](https://i.imgur.com/muJQ04p.png)
 
### Metric Learning

This idea of "reducing the complexity of the algorithm to reduce the need for data" is also used in the second family of approaches we will review, Metric Learning. This idea was proposed by Vinayls et al. in [*Matching Networks for One-Shot Learning*](https://arxiv.org/pdf/1606.04080.pdf) (2016). The authors aimed to switch from a Computer Vision space, where the problem is and which is typically solved through Deep Learning to another space more likely to be solved by non-parametric approaches (which don't need further training, e.g. kNN). To do so, they just train an embedding Network with attention, where attention acts as the final non-parametric matching algorithm (relates a given test image to each training one, related to one class each since we are working with One-shot Learning). The encoder network they propose is a RNN, the attention they use is Cosine Similarity and the loss they train with is a log one at Learning level while this is projected to the Meta-Learning level. This idea was also followed in [*Prototypical Networks for Few-Shot Learning*](https://arxiv.org/pdf/1703.05175.pdf) in 2017, where Snell et al. proposed a similar pipeline but instead of using a kNN-like algorithm (which Vinyals's attention mechanism stands for), they use a soft view of it, being each class in the space a Gaussian distribution instead of a discrete frontier. Each class distribution is called prototype in this paper, and it allows to use more than one sample per class, thus extending the problem to a Few-shot Learning problem instead of a One-shot one. The embedding function is learned from minimizing the negative log probability of the true class of the test samples. This approach of Meta-Learning is still popular nowadays due to its simplicity, and usually a good first step to experiment with Meta-Learning.
  
![Imgur](https://i.imgur.com/Yi60wou.png)
  
![Imgur](https://i.imgur.com/hSDCqyd.png)

### Optimizer Meta-Learning
  
Another way to achieve Meta-Learning recovers the old idea that Schmidhuber played with, learning an optimal optimizer (do you remember?). I think this idea need no longer presentation. Andrychowicz et al. presented in 2016 [*Learning to Learn By Gradient Descent By Gradient Descent*](https://arxiv.org/pdf/1606.04474.pdf), which is not a typo but its true name. Does this sound to you? If it doesn't, I'll remind you the Hochreiter publication in 2001 called Learning to Learn Using Gradient Descent. The idea is pretty similar. As well as you have an inner algorithm to solve the task, you also have an outer training algorithm. This training algorithm may also be optimized by training at the Meta-Learning level. So Andrychowicz calls them optimizee and optimizer. The optimizee is a parametric algorithm so it is actually optimized by its parameters (called Meta-parameters). Wait, isn't this the same than Hochreiter proposed? This gave me a bit of confusion and to be honest is one of the things I'm less sure about in the whole Meta-Learning topic. But my interpretation is that Hochreiter's idea was just a generalization that allowed to use gradient descent in both the optimizer and the optimizee, while this paper present architectures for specifically that. However, the most important idea for you here is that this view is still one of the main approaches of Meta-Learning. Later on, Larochelle et al. presented in 2017 the publication [*Optimization as a model for Few-Shot Learning*](https://openreview.net/pdf?id=rJY0-Kcll), which builds on the same idea but in this case the optimizer is the Gradient Descent itself, and instead of modifying Meta-Parameters acts as a weight predictor. The other important publication about that was done by Mishra et al in 2018, called [*A simple Neural Attentive Meta-Learner*](https://arxiv.org/pdf/1707.03141.pdf), where they extend the idea of Andrychowicz by instead of an RNN using an (soft) Attentional NN as the optimizer of Meta-parameters.

![Imgur](https://i.imgur.com/ZXAsypi.png)
  
![Imgur](https://i.imgur.com/MN8bSAL.png)
  
![Imgur](https://i.imgur.com/AVs0R5k.png)
  
### Initialization Meta-Learning

However, the probably most popular approach of Meta-Learning is the one presented by Finn et al. in [*Model-Agnostic Meta-Learning*](https://arxiv.org/pdf/1703.03400.pdf) for Fast Adaptation of Deep Networks (2017). These authors have made HUGE contributions to Meta-Learning, but this was the most iconic one. There, they presented MAML, a popular algorithm that aims to find a proper initialization for the whole domain of tasks.  This is applicable to any combination of parameters, thus becoming (as the title says) Model-Agnostic. At the Meta-Learning level, weights follow a path guided by a batch of tasks at each meta-step, where at each task the model learns and gives a final loss (after the desired few updates). This way it becomes able to (desirably) learn quickly when facing a new task. This algorithm is also pretty popular in Meta-Reinforcement Learning. But aside from MAML, we also have Reptile, presented by Nichol et al. in [*On First-Order Meta-Learning Algorithms*](https://arxiv.org/pdf/1803.02999.pdf) (2018), where basically the Meta-Learning trajectory follows also the individual tasks Learning one (what they find out to be the optimal path). Furthermore, Finn's team also presented in 2018 [*Probabilistic Model-Agnostic Meta-Learning*](https://arxiv.org/pdf/1806.02817.pdf) (Probabilistic MAML), while Kim et al. presented [*Bayesian Model-Agnostic Meta-Learning*](https://arxiv.org/pdf/1806.03836.pdf) (BMAML). To be honest, I'm not sure about the conceptual difference between both, but the conclusion is that the flexibility of MAML allowed even to introduce uncertainty, where the learned (and therefore initialized in MAML) weights worked in probabilistic frameworks, thus being distributions. Also Finn's team (again) presented [*Online Meta-Learning*](https://arxiv.org/pdf/1902.08438.pdf) (2019), where they used MAML in an Online scenario, where tasks were presented in a manner in which no information about future tasks was available at each batch.

![Imgur](https://i.imgur.com/6f3Fw6i.png)

![Imgur](https://i.imgur.com/cLdIkro.png)

![Imgur](https://i.imgur.com/XJ2Y8cG.png)

![Imgur](https://i.imgur.com/fpWeEt4.png)

![Imgur](https://i.imgur.com/wX4paDu.png)
  
### Modular Meta-Learning
  
Last, in 2020 Chen et al. published [*Modular Meta-Learning with Shrinkage*](https://arxiv.org/pdf/1909.05557.pdf), where they referred to what I think was the last of the big Meta-Learning approaches. They formalized the popular procedure when pretraining + fine-tuning is done by modules (e.g. the typical frozen backbone while fine-tuning heads). What they proposed is (meta-)learning the priors in which each module has to shrink (i.e. the strength to adapt to the task training). This way, they opened a door for new publications.
  
![Imgur](https://i.imgur.com/qUL4xKq.png)

### Summary
  
So, summarizing, the main strategies proposed to perform Meta-Learning are:

* Usage of memories
* Metric Learning (converting to non-parametric algorithm)
* Optimizer Learning
* Initialization Learning
* Modular Meta-Learning

## **Meta-Learning interesting uses**

* Few-Shot Learning: the first motivation of this wave of Meta-Learning publications. Learning from few data becomes possible when you learn how to learn with few data.
* Active Learning: I'm planning another post for this topic, but the problem stands for learning when human supervision has some costs. Again, possible if you learn how to solve this kind of problem.
* Unsupervised Learning: this is an interesting matter since we have defined everything under a supervised view, i.e. assuming we are able to design a Meta-Learning pipeline depending on a domain we will define usually knowing the classes. However, when we miss this information, we may find for an alternative way to define this schedule. In this direction, Metz et al. presented [*Meta-Learning Update Rules for Unsupervised Representation Learning*](https://arxiv.org/pdf/1804.00222.pdf) (2019), where they propose a way to achieve that by finding an (unknown) class space from which to build artificial tasks and train the model from them, thus projecting it to the new unsupervised tasks. The authors perform several experiments with the different main Meta-Learning approaches.

## **The future of Meta-Learning**

No, I'm not a prophet, but I have some ideas on what would be the natural direction of all this.

First, Meta-Learning has already presented the most intuitive approaches to be performed. As they still can be improved (just like any approach has received publications responding to it), the most important work there is presumably done.

However, Meta-learning just began. All Machine Learning problems that may potentially be affected by the (at least temporal) amount of data, may follow a Meta-Learning strategy. Probably, the Online scenario will gain strength in Meta-Learning research in the following years, since a common case is that a project begins with few data and later adds more and more. Furthermore, building good schedules is still not accomplished, so Curriculum Learning and Unsupervised scenarios will study in depth the application of Meta-Learning, for sure. Also, Meta-Learning still has a lot to say in other kinds of data limitations such as Incremental Learning (temporal bias issues), Active Learning (annotation issues), Federated Learning (privacy issues)...

Last, but not less important, Meta-Learning still has to deliver strong frameworks and stable implementations so it becomes more and more popular. So, congratulations for reading this post and preparing for the future!

## **Additional resources**

To complete this recap, I'm including a couple of summaries I did some time ago in two formats. First, a slide presentation which may be useful for a shorter [summary](https://drive.google.com/file/d/12xTctbkXcOHNX-ZtTA3ZaKUEi5Ulj_vc/view?usp=sharing). Second, a sheet with [a collection of important papers and notes](https://docs.google.com/spreadsheets/d/1IcaGSqPEVuF8iHD5G2wfl8xwpJmVuvnDOwrkZHIK1IU/edit?usp=sharing).

## **Thank you reader**

This is my first post, and writing it has been tough and has given me more work that I initially thought. However, the experience has filled me with more interest in continue making this blog live. As far as my life permits me to do it, I will be adding more content. This has just began!

