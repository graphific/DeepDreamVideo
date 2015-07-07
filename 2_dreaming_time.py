#!/usr/bin/python
__author__ = 'graphific'

import argparse
import os, os.path
import errno

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
from IPython.display import clear_output, Image, display

import caffe
caffe.set_mode_gpu()


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


#Load DNN
model_path = 'caffe/models/bvlc_googlenet/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


#Make dreams
def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size / np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', disp=False, clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis * (255.0 / np.percentile(vis, 99.98))
            if disp:
                showarray(vis)
            print octave, i, end, vis.shape
            if disp:
                clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# own functions
def morphPicture(filename1,filename2):
    img1 = PIL.Image.open(filename1)
    img2 = PIL.Image.open(filename2)
    return PIL.Image.blend(img1, img2, 0.5)

layersloop = ['inception_4c/output', 'inception_4d/output',
              'inception_4e/output', 'inception_5a/output',
              'inception_5b/output', 'inception_5a/output',
              'inception_4e/output', 'inception_4d/output',
              'inception_4c/output']



def make_sure_path_exists(path):
    '''
    make sure input and output directory exist, if not create them.
    If another error (permission denied) throw an error.
    '''
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main(input, output, disp, gpu):
    make_sure_path_exists(input)
    make_sure_path_exists(output)

    # should be picked up by caffe by default, but just in case
    # add by macpod
    if gpu:
        caffe.set_mode_gpu();
        caffe.set_device(0);
        
    frame = np.float32(PIL.Image.open(input+'/0001.jpg'))
    frame_i = 1
    
    # let max nr of frames
    nrframes =len([name for name in os.listdir('./input') if os.path.isfile(name)])

    for i in xrange(frame_i,nrframes):
        frame = deepdream(
            net, frame, end = layersloop[frame_i % len(layersloop)], disp=disp, iter_n=5)
        saveframe = output + "/%04d.jpg" % frame_i
        PIL.Image.fromarray(np.uint8(frame)).save(saveframe)
        newframe = input + "/%04d.jpg" % frame_i
        frame = morphPicture(saveframe, newframe) # give it back 50% of original picture
        frame = np.float32(frame)
        frame_i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dreaming in videos.')
    parser.add_argument(
        '-i','--input', help='Input directory where extracted frames are stored', required=True)
    parser.add_argument(
        '-o','--output', help='Output directory where processed frames are to be stored', required=True)
    parser.add_argument(
        '-d', '--display', help='display frames', action='store_false', dest='display')
    parser.add_argument(
        '-g', '--gpu', help='Use GPU', action='store_true', dest='gpu')
    args = parser.parse_args()
    
    if args.display:
        print("display turned on")

    main(args.input, args.output, args.display, args.gpu)

