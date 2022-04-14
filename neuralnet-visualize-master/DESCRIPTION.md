# Neural Network Visualizer

## General Description

A module which creates a neural network image with the given architecture. It's a handy tool to see how your network is built as compared to a model summary.

## Installation

### Before installation

Before installing the module, run the below command at your prompt to install the graphviz
```shell
$ sudo apt install graphviz
```

### Normal installation
```shell
$ sudo pip3 install neuralnet-visualize
```

### Development installation
```shell
$ git clone https://github.com/AnuragAnalog/nn_visualize.git
$ cd nn_visualize
```

### After installation

After installing the module, if you want to upgrade the module, run the below command.
```shell
sudo pip3 install neuralnet-visualize --upgrade
```

## Future Works

- [x] Add Convolutional layers, Maxpooling, Flatten layers
- [ ] Add Sequence model layers
- [ ] Directly from the pickle files
- [ ] Specific colors for activation functions
- [ ] Directly convert from pytorch models