import ImageFromClass as ifc
import numpy as np
import PIL.Image

ifc.use_gpu()
net = ifc.load_model()

octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':190,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':11.,
        'end_step_size':11.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.78,
        'start_step_size':6.,
        'end_step_size':6.
    },
    {
        'layer':'loss2/classifier',
        'scale':1.2,
        'iter_n':150,
        'start_sigma':0.78*1.2,
        'end_sigma':0.44,
        'start_step_size':6.,
        'end_step_size':3.
    },
    {
        'layer':'loss1/classifier',
        'iter_n':10,
        'start_sigma':0.44,
        'end_sigma':0.304,
        'start_step_size':3.,
        'end_step_size':3.
    }
]

# octaves = [
#     {
#         'layer':'loss3/classifier',
#         'iter_n':190,
#         'start_sigma':2.5,
#         'end_sigma':0.78,
#         'start_step_size':11.,
#         'end_step_size':11.
#     },
#     {
#         'layer':'loss3/classifier',
#         'scale':1.2,
#         'iter_n':450,
#         'start_sigma':0.78*1.2,
#         'end_sigma':0.40,
#         'start_step_size':6.,
#         'end_step_size':3.
#     }
# ]

#Change the color palette. 100, 100, 50 makes more greens and browns, less blues and whites
#lower numbers give darker colors
background_color = np.float32([100.0, 100.0, 100.0])
# generate initial random image
gen_image = np.random.normal(background_color, .001, (228, 228, 3))

#????img = np.float32(PIL.Image.open('youandmeSmall.jpg'))

for i in range(4, 1000):
    ifc.deepdraw(net, gen_image, octaves, i)