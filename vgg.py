import ImageFromClass as ifc
import numpy as np
import setup_caffe_network as su
import models as ml

octaves = [
    {
        'layer': 'fc7',
        'iter_n': 20,
        'start_sigma': 0.5,
        'end_sigma': 0.0,
        'start_step_size': 11.,
        'end_step_size': 3.
    }
]

#su.SetupCaffe.gpu_on() VGG won't fit on my 2 GB card
net = ml.NetModels.setup_vgg('')

# Change the color palette. 100, 100, 50 makes more greens and browns, less blues and whites
# lower numbers give darker colors
background_color = np.float32([255.0, 255.0, 255.0])
# generate initial random image
gen_image = np.random.normal(background_color, .001, (224, 224, 3))

IFC = ifc.ImageFromClass()
for i in range(2, 3):
    IFC.deepdraw(net, gen_image, octaves, i)
