#!/usr/bin/python3

from neuralnet_visualize import visualize as nnviz

network = nnviz.visualizer()

network.add_layer('dense', 9)
network.add_layer('dense', 5)

network.visualize()