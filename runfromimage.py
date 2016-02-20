import ImageFromClass as ifc
import models as ml
import cv2
import images


octaves = [
    {
        'layer': 'loss1/classifier',
        'iter_n': 10,
        'start_sigma': 0,
        'end_sigma': 0.0,
        'start_step_size': 6.,
        'end_step_size': 1.
    }
]

net = ml.NetModels.setup_googlenet_model('models/')

gen_img = cv2.imread('../ChiayiDreams/ImagesIn/20160212_141245.jpg')
gen_img = images.Images.resize_image(500, 500, gen_img)


IFC = ifc.ImageFromClass()
for i in range(0, 5):
    IFC.deepdraw(net, gen_img, octaves, i)
