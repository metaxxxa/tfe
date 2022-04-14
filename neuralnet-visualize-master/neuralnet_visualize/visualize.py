#!/usr/bin/python3

import graphviz as gv

from typing import Union

from .exceptions import *

class visualizer():
    """
    Neural Network Visualizer class

    Parameters
    ----------
    title : str, optional
        Title of your neural network. Default is "Neural Network"
    filename : str, optional
        Name of the image file. Default is "neuralnet"
    file_type : str, optional
        Extension of the file to be stored in, one of 'png', 'jpeg', 'jpg', 'svg', 'gif' (case-insensitive). Default is 'png'
    savepdf : bool, optional
        To save in pdf. Default is False
    orientation : str, optional
        orientation of the network architecture, one of 'LR', 'TB', 'BT', 'RL' (case-insensitive)

        LR means Left to Right, TB means Top to Bottom, BT means Bottom to Top, RL means Right to Left. Default is 'LR'

    Attributes
    ----------
    orient_ : str
        Orientation of the Architecture
    layers_ : int
        Total number of layers in the architecture
    network_ : graphviz.dot.Graph
        The Graphviz graph object
    layer_names_ : list
        Layer names in the architecture
    layer_types_ : list
        Layer types of the architecture
    layer_units_ : list
        Number of units each layer of the network
    nontrain_layers_ : int
        Number of layers whose parameters are non-trainable such as maxpool, avgpool, flatten etc.
    from_pytorch_called_ : bool
        To check wheather from_pytorch method called
    from_tensorflow_called_ : bool
        To check wheather from_tensorflow method called 

    Methods
    -------
    add_layer()
        Add a layer to the network
    from_pytorch()
        Make a network from the PyTorch model object
    from_tensorflow()
        Make a network from the TensorFlow model object
    get_meta_data()
        Return a dictionary containing the networks meta data
    summarize()
        Print a network summary in a MySQL Tabular format
    visualize()
        Render the network and open it with a suitable application

    Raises
    ------
    NotAValidOption
        When the option is not available

    Examples
    --------
    >>> from neuralnet_visualize import visualize as nnviz
    >>>
    >>> network = nnviz.visualizer()
    >>>
    >>> network.add_layer('dense', 7)
    >>> network.add_layer('dense', 12)
    >>> network.add_layer('dense', 4)
    >>>
    >>> network.visualize()
    """

    def __init__(self, title="Neural Network", filename='neuralnet', file_type='png', savepdf=False, orientation='LR'):
        self.title = title
        self.filename = filename
        self.color_encoding = {'input': 'yellow', 'hidden': 'green', 'output': 'red', 'conv2d': 'pink', 'maxpool2d': 'blue', 'avgpool2d': 'cyan', 'flatten': 'brown'}
        self.possible_layers = ['dense', 'conv2d', 'maxpool2d', 'avgpool2d', 'flatten', 'linear'] #, 'relu', 'sigmoid', 'softmax', 'swish']
        self.spatial_layers = ['conv2d', 'maxpool2d', 'avgpool2d', 'flatten']
        self.possible_filetypes = ['png', 'jpeg', 'jpg', 'svg', 'gif']
        self.possible_orientations = ['LR', 'TB', 'BT', 'RL']

        if savepdf:
            self.file_type = 'pdf'
        else:
            if file_type.lower() not in self.possible_filetypes:
                raise NotAValidOption(file_type, self.possible_filetypes)
            self.file_type = file_type

        if orientation.upper() not in self.possible_orientations:
            raise NotAValidOption(orientation, self.possible_orientations)
        self.orient_ = orientation

        self.network_ = gv.Graph(filename=filename, directory='./graphs', format=self.file_type,
              graph_attr=dict(ranksep='2', rankdir=self.orient_, label=title, labelloc='t', color='white', splines='line'),
              node_attr=dict(label='', nodesep='4', shape='circle', width='0.5'))

        self.layers_ = 0
        self.nontrain_layers_ = 0
        self.layer_names_ = list()
        self.layer_types_ = list()
        self.layer_units_ = list()
        self.from_torch_called_ = False
        self.from_tensorflow_called_ = False

    def __str__(self):
        return self.title

    def _check_dtype(self, value, val_type):
        # Check the datatype of the variable

        if isinstance(value, Union[list, tuple].__args__):
            if isinstance(value[0], int) and isinstance(value[1], int):
                if val_type == 'kernel_size':
                    vstr = "x".join(map(str, value))
                else:
                    vstr = str(tuple(value))
            else:
                raise TypeError("Expects a list/tuple of 2 integers")
        elif isinstance(value, int):
            if val_type == 'kernel_size':
                vstr = str(value)+"x"+str(value)
            else:
                vstr = str(value)
        else:
            raise TypeError("Expects an int or a list/tuple of 2 integers")

        return vstr

    def _update_meta_data(self, layer_type:str, nodes)->str:
        # Update the meta data of the network

        if layer_type.lower() in ['dense', 'linear']:
            self.layer_units_.append(nodes)
        elif layer_type.lower() in self.spatial_layers:
            self.layer_units_.append(1)

        if self.layers_ == 0:
            layer_name = layer_type.capitalize()+'_input'
        else:
            if layer_type.lower() in ['dense', 'conv2d']:
                layer_name = layer_type.capitalize()+'_hidden'+str(self.layers_ - self.nontrain_layers_)
            else:
                self.nontrain_layers_ = self.nontrain_layers_ + 1

                layer_name = layer_type.capitalize()+'_'+str(self.nontrain_layers_)

        self.layer_names_.append(layer_name)
        self.layer_types_.append(layer_type)
        self.layers_ = self.layers_ + 1

        return layer_name

    def add_layer(self, layer_type:str, nodes=10, filters=32, kernel_size=3, padding='valid', stride=1, pool_size=2)->None:
        """Adds a layer to the network

        Parameters
        ----------
        layer_type : str
            Type of layer to add to the network (case-insensitive)

            One of the 'dense', 'conv2d', 'maxpool2d', 'avgpool2d', 'flatten'.
        nodes : int, optional
            Number of units in the layer. Default is 10
        filters : int, optional
            Number of Kernels to be applied, only if layer_type == 'conv2d'. Default is 32
        kernel_size : int, tuple, list, optional
            Size of the 2D Convolution window, an integer or tuple/list of 2 integers only if layer_type == 'conv2d'. Default is 3
        padding : str, optional
            One of 'same' or 'valid' (case-insensitive), only if layer_type == 'conv2d'. Default is 'valid'
        stride : int, tuple, list, optional
            Stride of the Convolution window, an integer or tuple/list of 2 integers, only if layer_type == 'conv2d'. Default is 1
        pool_size : int, tuple, optional
            Size of the Maxpooling layer, an integer or tuple of 2 integers only if layer_type in ['maxpool2d', 'avgpool2d']. Default is 2

        Raises
        ------
        TypeError
            When the datatype of variable is not integer or list/tuple of two integers
        CannotCreateModel
            When a model cannot be created under certain conditions
        NotAValidOption
            When the layer_type is not implemented
        """

        if self.from_tensorflow_called_:
            raise CannotCreateModel("Network was already created from the tensorflow model object")
    
        if self.from_torch_called_:
            raise CannotCreateModel("Network was already created from the pytorch model object")

        if layer_type not in self.possible_layers:
            raise NotAValidOption(layer_type, self.possible_layers)

        layer_name = self._update_meta_data(layer_type, nodes)
        color = self.color_encoding.get(self.layer_types_[-1], 'black')

        if self.layer_types_[-1] == 'dense' or self.layer_types_[-1] == 'linear':
            with self.network_.subgraph(name='cluster_{}'.format(layer_name)) as layer:
                # update label so that title doesn't print multiple times
                layer.graph_attr.update(dict(label=""))

                if nodes > 10:
                    layer.attr(labeljust='right', labelloc='bottom', label='+'+str(nodes - 10))
                    nodes = 10

                for i in range(nodes):
                    if self.layers_ == 1:
                        color = self.color_encoding['input']
                    else:
                        color = self.color_encoding['hidden']
                    layer.node('{}_{}'.format(layer_name, i), shape='point', style='filled', fillcolor=color)
        elif self.layer_types_[-1] == 'conv2d':
            ksstr = self._check_dtype(kernel_size, 'kernel_size')
            sstr = self._check_dtype(stride, 'stride')

            content = "Kernal Size: "+ksstr+"\nFilters: "+str(filters)+"\nPadding: "+str(padding).capitalize()+"\nStride: "+sstr

            with self.network_.subgraph(node_attr=dict(shape='box3d')) as layer:
                layer.node(name=self.layer_names_[-1], label=content, height='1.5', width='1.5', style='filled', fillcolor=color)
        elif self.layer_types_[-1] == 'maxpool2d':
            pstr = self._check_dtype(pool_size, 'pool_size')
            content = "Max Pooling\nPool Size: "+pstr

            with self.network_.subgraph(node_attr=dict(shape='ellipse')) as layer:
                layer.node(name=self.layer_names_[-1], label=content, height='2', width='0.5', style='filled', fillcolor=color)
        elif self.layer_types_[-1] == 'avgpool2d':
            pstr = self._check_dtype(pool_size, 'pool_size')
            content = "Avg Pooling\nPool Size: "+pstr

            with self.network_.subgraph(node_attr=dict(shape='ellipse')) as layer:
                layer.node(name=self.layer_names_[-1], label=content, height='2', width='0.5', style='filled', fillcolor=color)
        elif self.layer_types_[-1] == 'flatten':
            with self.network_.subgraph(node_attr=dict(shape='rectangle')) as layer:
                layer.node(name=self.layer_names_[-1], label='Flatten', height='4.5', width='0.5', style='filled', fillcolor=color)

        return

    def _connect_layers(self, l1_nodes, l2_nodes, l1_idx, l2_idx):
        # Connect all the nodes between the two layers

        for l1 in range(l1_nodes):
            for l2 in range(l2_nodes):
                if self.layer_types_[l1_idx] in ['dense', 'linear'] and self.layer_types_[l2_idx] in ['dense', 'linear']:
                    n1 = self.layer_names_[l1_idx]+'_'+str(l1)
                    n2 = self.layer_names_[l2_idx]+'_'+str(l2)
                elif self.layer_types_[l1_idx] in ['dense', 'linear'] and self.layer_types_[l2_idx] in self.spatial_layers:
                    n1 = self.layer_names_[l1_idx]+'_'+str(l1)
                    n2 = self.layer_names_[l2_idx]
                elif self.layer_types_[l1_idx] in self.spatial_layers and self.layer_types_[l2_idx] in ['dense', 'linear']:
                    n1 = self.layer_names_[l1_idx]
                    n2 = self.layer_names_[l2_idx]+'_'+str(l2)
                elif self.layer_types_[l1_idx] in self.spatial_layers and self.layer_types_[l2_idx] in self.spatial_layers:
                    n1 = self.layer_names_[l1_idx]
                    n2 = self.layer_names_[l2_idx]

                self.network_.edge(n1, n2)

        return

    def _build_network(self):
        # Connect all the adjacent layers in the network

        for i in range(self.layers_ - 1):
            nodes1 = self.layer_units_[i]
            nodes2 = self.layer_units_[i+1]

            if self.layer_units_[i] > 10:
                nodes1 = 10
            if self.layer_units_[i+1] > 10:
                nodes2 = 10

            self._connect_layers(nodes1, nodes2, i, i+1)

        if self.layer_types_[-1] in ['dense', 'linear']:
            # Updating the color of output dense layer to red

            nodes = ((self.layer_units_[-1] > 10) and 10) or self.layer_units_[-1]
            with self.network_.subgraph(name='cluster_{}'.format(self.layer_names_[-1])) as layer:
                for i in range(nodes):
                    layer.node('{}_{}'.format(self.layer_names_[-1], i), style='filled', fillcolor='red')

        return

    def from_pytorch(self, model):
        """Converts a given PyTorch model into graph

        Parameters
        ----------
        model : torch.nn.modules.container.Sequential
            A pytorch model

        Raises
        ------
        ValueError
            Multiple calls when it is already initialized.
        """

        if self.from_torch_called_ == True:
            raise ValueError("The model has already been initialised, with a PyTorch model")

        layers = model.modules() # Returns a generator object
        split = str.split

        for layer in layers:
            layer_name = split(str(type(layer)),"'")[1] # Returns string like torch.nn.modules.container.Sequential/layer
            layer_name = split(layer_name,'.')[-1].lower() # Splitting by . gives us name of layer

            if layer_name not in self.possible_layers:  # Skip specific activation layers
                continue

            self.add_layer(**self._create_dict(layer,layer_name))

        self.from_torch_called_ = True

        return

    def _create_dict(self, layer, layer_name):
        # creates a parameter dict of layer

        params = {}

        params['layer_type'] = layer_name

        if layer_name == 'conv2d':
            params['kernel_size'] = layer.kernel_size
            params['filters'] = layer.out_channels
            params['stride'] = layer.stride
            params['padding'] = layer.padding
        elif layer_name in ['maxpool2d', 'avgpool2d']:
            params['pool_size'] = layer.kernel_size
        elif layer_name == 'linear':
            params['layer_type'] = 'dense'
            params['nodes'] = layer.out_features

        return params

    def from_tensorflow(self, model):
        """Converts a given TensorFlow model into graph

        Parameters
        ----------
        model : tensorflow.python.keras.engine.sequential.Sequential
            A tensorflow model
        """

        layers = model.get_config()['layers']
        for layer in layers:
            layer_params = layer['config']
            if layer['class_name'] == 'Dense':
                self.add_layer('dense', nodes=layer_params['units'])
            elif layer['class_name'] == 'Conv2D':
                self.add_layer('conv2d', filters=layer_params['filters'], kernel_size=layer_params['kernel_size'], stride=layer_params['strides'], padding=layer_params['padding'])
            elif layer['class_name'] == 'MaxPooling2D':
                self.add_layer('maxpool2d', pool_size=layer_params['pool_size'])
            elif layer['class_name'] == 'AveragePooling2D':
                self.add_layer('avgpool2d', pool_size=layer_params['pool_size'])
            elif layer['class_name'] == 'Flatten':
                self.add_layer('flatten')

        self.from_tensorflow_called_ = True

        return

    def set_color_encoding(self, encoding):
        """Set Custom Color Encoding to the layers

        Parameters
        ----------
        encoding : dict
            Dictionary which contains the color encoding of the layers

        Raises
        ------
        ValueError
            When the passed object is not a dict
        """

        if not isinstance(encoding, dict):
            raise ValueError("Expected a dict, but got {}".format(type(encoding)))

        for k, _ in encoding:
            if self.color_encoding.get(k, False):
                self.color_encoding[k] = encoding[k]

        return

    def get_meta_data(self):
        """Give a dictionary which contains meta data of the network.

        Returns
        -------
        meta_data : dict
            meta data which contains the details of all the layerss
        """

        meta_data = dict()

        meta_data['Number of Layers'] = self.layers_
        meta_data['Layer names'] = self.layer_names_
        meta_data['Layer Types'] = self.layer_types_
        meta_data['Node in Layers'] = self.layer_units_

        return meta_data

    def summarize(self):
        """Prints a summary of the network in MySQL tabular format.\nCurrently, we are support tensorflow 
        models.\n We will implement pytorch summarization soon

        Raises
        ------
        ValueError
            when the model is not yet created
        """

        if not self.from_torch_called_ and not self.from_tensorflow_called_ and self.layers_ < 2:
            if not self.from_torch_called_ and not self.from_tensorflow_called_:
                raise ValueError('This model has not yet been created. Create the model first by calling `from_pytorch()` or calling `from_tensorflow()`')
            else:
                raise ValueError('The model has not been built yet or the model is not supported.\n Check the docs for further information')
        
        title = "Neural Network Architecture"
        hline = "+"+"-"*69+"+"

        print(hline)
        print("|"+title.center(69)+"|")
        print(hline)
        print("|"+"Layer Name".center(28)+"|"+"Layer Type".center(24)+"|"+"Layer Units".center(15)+"|")
        print(hline)
        for i in range(self.layers_):
            col1 = self.layer_names_[i].center(28)
            col2 = self.layer_types_[i].capitalize().center(24)
            col3 = str(self.layer_units_[i]).center(15)
            print("|"+col1+"|"+col2+"|"+col3+"|")
            print(hline)

        return

    def visualize(self, give_obj=False):
        """Visualize the network

        Opens an image containing the visualised network

        Parameters
        ----------
        give_obj : bool, optional
            If set true, returns the graph object. Default is False

        Returns
        -------
        network_ : graphviz.dot.Graph
            The graphviz graph object after complete building of the network

        Raises
        ------
        CannotCreateModel
            When a model cannot be created under certain conditions
        """

        if self.layers_ < 2:
            raise CannotCreateModel("Cannot draw Neural Network, Add atleast two layers to the network")

        self._build_network()

        if give_obj:
            return self.network_

        self.network_.view()

        return

if __name__ == '__main__':
    input_nodes = 7
    hidden_nodes = 12
    output_nodes = 4

    net = visualizer()

    # net.add_layer('conv2d')
    # net.add_layer('dense', hidden_nodes)
    # net.add_layer('dense', output_nodes)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='sigmoid'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation='sigmoid'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid')
    ])

    net.from_tensorflow(model)
    net.visualize()
    net.summarize()