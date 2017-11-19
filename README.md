# Learning image processing with neural network

This project aims at giving a common platform for learning image processing 
task with neural network. The usage of the module is very simple:

1. Specify an image processing pipeline using traditional Python code
2. Declare a neural network model
3. Train the network with pre/post-processing methods
4. Save and test the model

Please read the following sections to understand how each component is
declared and how to contribute to the project.

## Components

The project specifies a couple of interfaces that standardizes the usage of
the module. In addition to the common interface, we provide some 
implementation that occur frequently, including the implementation of  
`Model` and `Trainer` in some popular backends (see Backends for more details).
most interfaces have

### Dataset

The `Dataset` interface is the I/O interface of the project and offers a unique 
way to store images, in a large array. A couple of common datasets are provided
such as `ImageFolder` and `ImageSingleton`. 

### Model

The `Model` interface is a simple wrapper around a neural network model. It 
doesn't specify the way the network will be trained or evaluated. Such
responsibilities are delegated to the `Trainer` object. However, it 
standardizes the model saving and loading methods. The loading method is not 
part of the module, but it outputs an object implementing `Model`. 

### Processors

If the above mentioned interfaces can be found in other frameworks, the 
processors are what is special about the project. There are 3 types of 
processors, each one of them is encapsulated by a callable object / interface.

1. `TargetProcessor`: the image processing task to imitate. It first takes some
parameters to initialize. Then it can be apply to any images and produces
input / output pair and some metadata reflecting the parameters such as the 
noise map. Please note that the input to `TargetProcessor` may not be the input
in the processing pipeline. For example, in case of denoising, the input image
actually corresponds to the output of the denoising process, while as the input
of the denoising process is a noisy version of the input image.
2. `BatchProcessor`: training data sampler. After we have our input / output 
image pair and the metadata, we need to feed them to the model. 
Depending on the model definition, it may take a full image, a patch or other
format. In order to satisfy a wide range of demand, the `BatchProcessor` is
introduced to convert a list of input / output pair and the metadata to a 
minibatch of training samples. The size of the minibatch, the shape are 
hardcoded in the object that implements `BatchProcessor`. 
3. `ModelProcessor`: wrapper around the model to imitate the target pipeline.
This processor has 2 components, one pre-processor that works similarly to
`BatchProcessor` which now takes one single image and metadata as input,
generates a list of data points that can be feed to the `Model` (no output
is needed here as we are not going to train the model). After processing 
these data points with `Model`, a post-processor is used to reassemble the
image in a predefined way.

### Trainer / Tester

The `Trainer` interface unifies the training and evaluation of the `Model`.
It takes as input a `Dataset`, a `Model`, some `Processors` and a 
`Parameters` that specify all parameters relevant to the network training such
as the learning rate and number of epochs. The object can be used in 2 phases:
training and testing phase.

The `Tester` module is not a interface, but a class. It builds upon a `Model`
and a `ModelProcessor` which control the whole computation. Therefore, `Tester`
is merely a wrapper around those 2 objects, but it does offer a unified
method for interacting with images or `Dataset`.

## Backends

Given the variety of neural network frameworks, the project supports a couple
of widely used ones. The support for other frameworks can be added easily
by implementing the `Model` and `Trainer` interface. The reason being those
frameworks use different APIs for network training and evaluation. Please note
that the definition of neural network is not included in the project, but 
we do give some examples under the samples folder.

- [x] Keras
- [ ] Tensorflow
- [ ] Chainer

## Examples

To install the project, run:

```
git clone https://github.com/lyx-x/nnimgproc.git
cd nnimgproc
pip3 install --user -e '.[all]'
```
In addition to Arch Linux under which the project is developed, the project
is also tested against the environment specified in the Dockerfiles under
`docker`. To build a docker image and run a container in the interactive mode,
please execute:

```
# build an cpu-lonimage

# use cpu-only
docker
# use gpu
nvidia-docker
```

In order to use GPUs, run:

```
nvidia-docker run -it --rm nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04 bash
```


### Image denoising with multilayer perceptron

Under the `samples/denoising` folder, we provide some scripts for training 
simple neural network-based image denoiser. The default parameters are taken
from Burger et al. 
[[1]](http://people.tuebingen.mpg.de/burger/neural_denoising/cvpr2012.html).
To train and test a smaller version of it, please use the supplemented
bash script.

```
export DATA=#{A folder containing enough images to be loaded}
bash samples/denoising/run.sh
```







