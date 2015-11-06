import ImageFromClass as ifc
import numpy as np

ifc.use_gpu()
net = ifc.load_model()

#scale doesn't work
#its 190 450
octaves = [
    {
        'layer':'loss3/classifier',
        'iter_n':100,
        'start_sigma':2.5,
        'end_sigma':0.78,
        'start_step_size':11.,
        'end_step_size':11.
    },
    {
        'layer':'loss3/classifier',
        'scale':1.2,
        'iter_n':200,
        'start_sigma':0.78*1.2,
        'end_sigma':0.40,
        'start_step_size':6.,
        'end_step_size':3.
    }
]
background_color = np.float32([100.0, 100.0, 50.0])
# generate initial random image
gen_image = np.random.normal(background_color, .001, (228, 228, 3))

ifc.deepdraw(net, gen_image, octaves)