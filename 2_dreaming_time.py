#!/usr/bin/python
__author__ = 'graphific'

import argparse
import os, os.path
import errno
import sys
import time

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))




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
            print(octave, i, end, vis.shape)
            if disp:
                clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def resizePicture(image,width):
	img = PIL.Image.open(image)
	basewidth = width
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	return img.resize((basewidth,hsize), PIL.Image.ANTIALIAS)

def morphPicture(filename1,filename2,blend,width):
	img1 = PIL.Image.open(filename1)
	img2 = PIL.Image.open(filename2)
	if width is not 0:
		img2 = resizePicture(filename2,width)
	return PIL.Image.blend(img1, img2, blend)

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

layersloop = ['inception_4c/output', 'inception_4d/output',
              'inception_4e/output', 'inception_5a/output',
              'inception_5b/output', 'inception_5a/output',
              'inception_4e/output', 'inception_4d/output',
              'inception_4c/output']
def main(input, output, disp, gpu, model_path, model_name, preview, octaves, octave_scale, iterations, jitter, zoom, stepsize, blend, layers):
    make_sure_path_exists(input)
    make_sure_path_exists(output)
    
     # let max nr of frames
    nrframes =len([name for name in os.listdir(input) if os.path.isfile(os.path.join(input, name))])
    if nrframes == 0:
        print("no frames to process found")
        sys.exit(0)
    
    if preview is None: preview = 0
    if octaves is None: octaves = 4
    if octave_scale is None: octave_scale = 1.5
    if iterations is None: iterations = 5
    if jitter is None: jitter = 32
    if zoom is None: zoom = 1
    if stepsize is None: stepsize = 1.5
    if blend is None: blend = 0.5
    if layers is None: layers = 'customloop' #['inception_4c/output']
    
    #Load DNN
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + model_name #'bvlc_googlenet.caffemodel'
    
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))
    
    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
                           
    if gpu == '0':
    	caffe.set_mode_gpu()
    	caffe.set_device(int(args.gpu))
    	print("GPU mode [device id: %s]" % args.gpu)
    	print("using GPU, but you'd still better make a cup of coffee")
	else: print("SHITTTTTTTTTTTTTT You're running CPU man =D")

    if disp:
        from IPython.display import clear_output, Image, display
        print("display turned on")
        
    frame = np.float32(PIL.Image.open(input+'/0001.jpg'))
    if preview is not 0:
        frame = resizePicture(input+'/0001.jpg',preview)
    frame_i = 1
    
    now = time.time()
    
    for i in xrange(frame_i,nrframes):
        print('Processing frame #{}').format(frame_i)
        
        if layers == 'customloop': #loop over layers as set in layersloop array
            endparam = layersloop[frame_i % len(layersloop)]
            frame = deepdream(
                net, frame, iter_n = iterations, step_size = stepsize, octave_n = octaves, octave_scale = octave_scale, 
                jitter=jitter, end = endparam)
        else: #loop through layers one at a time until this specific layer
            endparam = layers[frame_i % len(layers)]
            frame = deepdream(
                net, frame, iter_n = iterations, step_size = stepsize, octave_n = octaves, octave_scale = octave_scale, 
                jitter=jitter, end = endparam)

        saveframe = output + "/%04d.jpg" % frame_i
        
        later = time.time()
        difference = int(later - now)
        # Stats (stolen + adapted from Samim: https://github.com/samim23/DeepDreamAnim/blob/master/dreamer.py)
        print '***************************************'
        print 'Saving Image As: ' + saveframe
        print 'Frame ' + str(i) + ' of ' + str(nrframes)
        print 'Frame Time: ' + str(difference) + 's' 
        timeleft = difference * (nrframes - frame_i)
        m, s = divmod(timeleft, 60)
        h, m = divmod(m, 60)
        print 'Estimated Total Time Remaining: ' + str(timeleft) + 's (' + "%d:%02d:%02d" % (h, m, s) + ')' 
        print '***************************************'
        
        PIL.Image.fromarray(np.uint8(frame)).save(saveframe)
        newframe = input + "/%04d.jpg" % frame_i
        
        if blend == 0:
            newimg = PIL.Image.open(newframe)
            if preview is not 0:
                newimg = resizePicture(newframe,preview)
            frame = newimg
        else:
            frame = morphPicture(saveframe,newframe,blend,preview)
                
        frame = np.float32(frame)
        
        now = time.time()
        frame_i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dreaming in videos.')
    parser.add_argument(
        '-i','--input', 
        help='Input directory where extracted frames are stored', 
        required=True)
    parser.add_argument(
        '-o','--output', 
        help='Output directory where processed frames are to be stored', 
        required=True)
    parser.add_argument(
        '-d', '--display', 
        help='display frames', 
        action='store_true', 
        dest='display')
    parser.add_argument(
        "--gpu",
        default='0',
        help="Switch for gpu computation."
    ) #int can chose index of gpu, if theres multiple gpus to chose from
    parser.add_argument(
        '-t', '--model_path', 
        dest='model_path', 
        default='../caffe/models/bvlc_googlenet/',
        help='Model directory to use')
    parser.add_argument(
        '-m', '--model_name', 
        dest='model_name', 
        default='bvlc_googlenet.caffemodel',
        help='Caffe Model name to use')
    parser.add_argument(
        '-p','--preview', 
        type=int, 
        required=False,
        help='Preview image width. Default: 0')
    parser.add_argument(
        '-oct','--octaves', 
        type=int, 
        required=False,
        help='Octaves. Default: 4')
    parser.add_argument(
        '-octs','--octavescale', 
        type=float, 
        required=False,
        help='Octave Scale. Default: 1.4',)
    parser.add_argument(
        '-itr','--iterations',
        type=int, 
        required=False,
        help='Iterations. Default: 10')
    parser.add_argument(
        '-j','--jitter', 
        type=int, 
        required=False,
        help='Jitter. Default: 32')
    parser.add_argument(
        '-z','--zoom', 
        type=int,
        required=False,
        help='Zoom in Amount. Default: 1')
    parser.add_argument(
        '-s','--stepsize', 
        type=float, 
        required=False,
        help='Step Size. Default: 1.5')
    parser.add_argument(
        '-b','--blend', 
        type=float, 
        required=False,
        help='Blend Amount. Default: 0.5')
    parser.add_argument(
        '-l','--layers',
        nargs="+", 
        type=str, 
        required=False,
        help='Array of Layers to loop through. Default: [customloop] \
        - or choose ie [inception_4c/output] for that single layer')
    
    args = parser.parse_args()
    
    if not args.model_path[-1] == '/':
        args.model_path = args.model_path + '/'
        
    if not os.path.exists(args.model_path):
        print("Model directory not found")
        print("Please set the model_path to a correct caffe model directory")
        sys.exit(0)
    
    model = os.path.join(args.model_path, args.model_name)
    
    if not os.path.exists(model):
        print("Model not found")
        print("Please set the model_name to a correct caffe model")
        print("or download one with ./caffe_dir/scripts/download_model_binary.py caffe_dir/models/bvlc_googlenet")
        sys.exit(0)

    main(
        args.input, args.output, args.display, args.gpu, args.model_path, args.model_name, 
        args.preview, args.octaves, args.octavescale, args.iterations, args.jitter, args.zoom, args.stepsize, args.blend, args.layers)

