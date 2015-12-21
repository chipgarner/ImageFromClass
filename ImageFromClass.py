# imports
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import cv2


class ImageFromClass:
    def __init__(self):
        pass

    # def load_image(image_relative_path):
    #     img = np.float32(PIL.Image.open(image_relative_path))
    #     return img

    # a couple of utility functions for converting to and from Caffe's input image layout
    def __preprocess(self, net, img):
        return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

    def __deprocess(self, net, img):
        return np.dstack((img + net.transformer.mean['data'])[::-1])

    labels = np.loadtxt('models/bvlc_googlenet/synset_words.txt', str, delimiter='\t')
    def __name_result_class(self, imageNetClass):

        return "output/" + str(imageNetClass) + " " + self.labels[imageNetClass] + ".jpg"

    def __save_result(self, path, vis):
        # adjust image contrast and clip
        vis = vis * (255.0 / np.percentile(vis, 99.98))
        vis = np.uint8(np.clip(vis, 0, 255))

        pimg = PIL.Image.fromarray(vis)

        pimg.save(path, 'jpeg')

    def __show_result(self, vis):
        # adjust image contrast and clip
        vis = vis * (255.0 / np.percentile(vis, 99.98))
        vis = np.uint8(np.clip(vis, 0, 255))

        pimg = PIL.Image.fromarray(vis)
        pimg.show()

        # showOpenCV(vis)

    def __showOpenCV(self, image):
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # I think this is being converted both ways ...
        cv2.imshow("test", bgr)
        cv2.waitKey(0)  # Scripting languages are weird, It will not display without this
        cv2.destroyAllWindows()

    def __blur(self, img, sigma):
        if sigma > 0:
            img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
            img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
            img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
        return img

    def objective_L2(dst):
        dst.diff[:] = dst.data

    def objective_L2_class(dst, imageNetClass=681):  # 366 gorrilla, 386 elephant, 834 suit, notebook 681, tandem 444

        one_hot = np.zeros_like(dst.data)
        one_hot.flat[imageNetClass] = 1.
        dst.diff[:] = one_hot

    def __make_step(self, net, netClass, end, sigma, step_size=3, objective=objective_L2_class):
        '''Basic gradient ascent step.'''

        src = net.blobs['data']  # input image is stored in Net's 'data' blob
        dst = net.blobs[end]

        net.forward(end=end)
        objective(dst, netClass)  # specify the optimization objective
        net.backward(start=end)

        g = src.diff[0]

        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = self.__blur(src.data[0], sigma)

    ###################################################

    def deepdraw(self, net, base_img, octaves, netClass):
        # reshape and load image
        source = net.blobs['data']
        h, w, c = base_img.shape[:]
        source.reshape(1, 3, h, w)
        source.data[0] = self.__preprocess(net, base_img)

        # self.__show_result(base_img)

        for e, o in enumerate(octaves):
            layer = o['layer']

            for i in xrange(o['iter_n']):
                sigma = o['start_sigma'] + ((o['end_sigma'] - o['start_sigma']) * i) / o['iter_n']
                step_size = o['start_step_size'] + ((o['end_step_size'] - o['start_step_size']) * i) / o['iter_n']

                self.__make_step(net, netClass, end=layer, sigma=sigma, step_size=step_size)

        # visualization
        vis = self.__deprocess(net, source.data[0])
        # self.__show_result(vis)

        self.__save_result(self.__name_result_class(netClass), vis)
