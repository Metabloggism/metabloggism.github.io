# **Short: ONNX**

This is the first **SHORT** post in Metabloggism. Feel free to contact me to give any feedback!

## **The questions**

Have you heard the name *ONNX*? If you have, do you understand the reason why it exists? In this Short I will try to cover the interest to any answer to the previous questions. For those who haven't heard it, read it as if I'm introducing you a tool that may or not be of interest (and that will be your decision). For those who have, but have different levels of knowledge about it, I pretend to cover the gaps in your knowledge and contextualize why you are hearing this. And for those who know it, maybe you are not the main target of the Short and you may decide to skip it, but if still you follow, take it as a review/contrast with your knowledge as well as an opportunity to think about alternative ways to take profit from it.

## **Definition of ONNX**

Instead of talking about others, why don't I let ONNX present itself? From the [official website](https://onnx.ai/):

> *ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers*

Ok, but the blog has the task of explaining it still. In my words, ONNX is a standard format for ML models that delivers a library with conversion tools to and from most of the main ML frameworks. It is also built thinking in users both experimenting and seeking optimization and therefore supports both training and runtime frameworks.

![ONNX](https://miro.medium.com/max/720/0*o5doPyWdduatUKtX.PNG)

## **Where does ONNX come from?**

ONNX was the result of a collaboration between big companies. Initially developed by PyTorch team (Facebook), Microsoft and AWS joined in different ways as Co-Founders. They announced it in 2017 as a step further toward a common goal: collaborative development of AI.

This is one key concept as it allows AI researchers/developers to exploit their advances by destroying some limitations given by the differences between frameworks, thus giving an easier access to the AI community. AI research and divulgation (like what this blog aims to do) also point in that direction.

Soon, almost all the main AI frameworks/platforms began giving support to ONNX.

![Bigtechs](https://i.imgur.com/eHeJ47l.png)

## **ONNX strategy**

To support interoperability, ONNX relies in instead of being a serialization method (like *JSON* or *pkl*), where a deserialization is mandatory and that could depend on the framework (in addition to the extra deserialization step resource consumption), directly stores the raw operations.

These raw operations are part of the *ONNX* standard in a collection called [*opset*](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

## **ONNX ML frameworks support**

The main (in development) ML frameworks support ONNX through some API. The following image ([source](https://onnx.ai/supported-tools.html#buildModel)) shows some of the frameworks that support ONNX:

![ML frameworks](https://i.imgur.com/MFt84AK.png)

Note that not only Neural Networks are included, but also boosting libraries like [XGBoost](https://xgboost.readthedocs.io/en/stable/) or [CatBoost](https://catboost.ai/), [SciKit-Learn](https://scikit-learn.org/stable/) (which includes several classical algorithms), ...

## **ONNX Runtime engines support**

ONNX supports different Runtime engines to allow the users to run the models on their own specific hardware. For example, one could have a Pytorch trained model and wants to include it in an iOS app, but that demands to use [CoreML](https://developer.apple.com/documentation/coreml). As CoreML is included in the supported frameworks, that could be done without issues through ONNX. That applies to any hardware with a runtime engine supporting ONNX. Do you want your model in a smart fridge? You will need a model runtime engine on the fridge. Just include support to ONNX and you will be able to run your PyTorch models.

However, ONNX itself is independent of other frameworks and includes [ONNX Runtime](https://onnxruntime.ai/), a toolset that allows to directly run the model.

Again, the image below shows some of these runtime frameworks:

![Runtime engines](https://i.imgur.com/URfGOgr.png)

## **ONNX model zoo**

There is also good news if you want even to skip your training phase and quickly run a model in your hardware. ONNX already offers some trained ONNX models in ONNX format in their [model zoo](https://github.com/onnx/models) so you can directly run it with ONNX Runtime (or the chosen Runtime engine).

## **Associated tools**

### Cloud services

The following Cloud engines give explicit support to ONNX models:

![Cloud services](https://i.imgur.com/VLydLF0.png)

### Visualization

The following tools allow you to inspect the models in a graphical way:

![Visualization tools](https://i.imgur.com/Guvmxwg.png)

In one section below I will show an example using [*Netron*](https://netron.app/).

## **ONNX example: from Pytorch**

I developed a Python snippet of code that converts [ResNet18 from torchvision](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) into ONNX format (and stores it in disk). It also includes a method to load the model from the file.

```python
import torch
import torch.onnx
import onnx
from torchvision.models import resnet18, ResNet18_Weights


def resnet18_to_onnx(fname='resnet18.onnx', batch_size=8):
    """
    Saves on disk in ONNX the ResNet18 pretrained model from Pytorch, which works through batches

    :param fname: (str)
    :param batch_size: (int) batch size of the input
    :return:
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    batch = torch.zeros([batch_size, 3, 256, 256])   # ImageNet1k size
    torch.onnx.export(model, batch, fname, input_names=['input'], output_names=['output'])


def load_resnet18_from_onnx(fname='resnet18.onnx'):
    return onnx.load(fname)


if __name__ == "__main__":
    resnet18_to_onnx()

```

### Visualization example

I loaded the ONNX model from the example in the app *Netron* and I could visualize the model in the following image:

![Imgur](https://i.imgur.com/lj4aiBU.png)

## Some ONNX limitation

Until now we only talked about how good is ONNX. I strongly believe it is as an initiative, but obviously we can have a conversation on how limited it is (as we could do with anything). We will not do so, but I will remark what I think is the main limitation of ONNX.

Although *opset* is pretty wide, we could want to use our custom layers/operations. And that is where ONNX fails us. ONNX only supports operations in the *opset*. In that case, we could find an alternative way to do what we want or just don't use ONNX.

## Conclusions

Although not 100% of the cases can be covered by ONNX, nearly all of them can. Nowadays, frameworks already offer tools to intuitively convert our models into ONNX and load them. Thus, I think we developers should adopt the policy to convert the models to ONNX for our own model zoos. These policies should obviously also consider the possibility of some models not being able to be converted to ONNX, allowing for different solutions in that cases.

## Thank you, reader

Thank you for going with me through this first Short post! I expect to do more like this in the future so any feedback is welcome.
