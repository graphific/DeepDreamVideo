# DeepDreamVideo
Implementing **#deepdream** on video

**Creative Request**

It would be very helpful for other deepdream researchers, if you could **include the used parameters in the description of your youtube videos**. You can find the parameters in the image filenames.

Included experiment: Deep Dreaming Fear & Loathing in Las Vegas: the Great Fan Francisco Acid Wave

The results can be seen on youtube: https://www.youtube.com/watch?v=oyxSerkkP4o

Mp4 not yet destroyed by youtube compression also at [mega.nz](https://mega.nz/#!KldUTKKD!38qj6WtEOE4pno90dAW98gkNK2O3tvz6ZwKTxpHJWFc) together with [original video file](https://mega.nz/#!X9MWWDTQ!lbC7C5B4incMkLGVM00qwI4NP-ifi2KcqsmfsdIm_E0).

All single processed + unprocessed frames are also at [github](https://github.com/graphific/Fear-and-Loathing-experiment)

![deepdreamanim1](http://media.giphy.com/media/l41lRx92QqsIXy5MI/giphy.gif "deep dream animation 1")
![deepdreamanim2](http://media.giphy.com/media/l41lSzjTsGJcIzpKg/giphy.gif "deep dream animation 2")

Advise also at https://github.com/graphific/DeepDreamVideo/wiki

##INSTALL Dependencies

A good overview (constantly being updated) on which software libraries to install & list of web resources/howto is at reddit: https://www.reddit.com/r/deepdream/comments/3cawxb/what_are_deepdream_images_how_do_i_make_my_own/


##Usage:

Extract 25 frames a second from the source movie

`./1_movie2frames.sh ffmpeg [movie.mp4] [directory]`

or

`./1_movie2frames.sh avconf [movie.mp4] [directory]`

Let a pretrained deep neural network dream on it frames, one by one, taking each new frame and adding 0-50% of the old frame into it for continuity of the hallucinated artifacts, and go drink your caffe

`python 2_dreaming_time.py -i frames -o processed --gpu`

different models can be loaded with:

`python 2_dreaming_time.py -i frames -o processed --model_path ../caffe/models/Places205-CNN/ --model_name Places205.caffemodel --gpu`

(again eat your heart out, Not a free lunch, but free models are [here](https://github.com/BVLC/caffe/wiki/Model-Zoo))

and sticking to one specific layer:

`python 2_dreaming_time.py -i frames -o processed -l inception_4c/output --gpu`

(**don't forget the --gpu flag if you got a gpu to run on**)

Once enough frames are processed (the script will cut the audio to the needed length automatically) or once all frames are done, put the frames + audio back together:

`./3_frames2movie.sh [frames_directory] [original_video_with_sound]`


##More information:

This repo implements a deep neural network hallucinating Fear & Loathing in Las Vegas. Visualizing the internals of a deep net we let it develop further what it think it sees.

We're using the #deepdream technique developed by Google, first explained in the Google Research blog post about Neural Network art.

- http://googleresearch.blogspot.nl/2015/06/inceptionism-going-deeper-into-neural.html

Code:

- https://github.com/google/deepdream


## parameters used (and useful to play with):

- network: standard reference GoogLeNet model trained on ImageNet from the Caffe Model Zoo (https://github.com/BVLC/caffe/wiki/Model-Zoo)

- iterations: 5

- jitter: 32 (default)

- octaves: 4 (default)

- layers locked to moving upwards from inception_4c/output to inception_5b/output (only the output layers, as they are most sensitive to visualizing "objects", where reduce layers are more like "edge detectors") and back again

- every next unprocessed frame in the movie clip is blended with the previous processed frame before being "dreamed" on, moving the alpha from 0.5 to 1 and back again (so 50% previous image net created, 50% the movie frame, to taking 100% of the movie frame only). This takes care of "overfitting" on the frames and makes sure we don't iteratively build more and more "hallucinations" of the net and move away from the original movie clip.


## An investigation of using the MIT Places trained CNN (mainly landscapes):

https://www.youtube.com/watch?v=6IgbMiEaFRY


## Installing DeepDream:

- original Google code is relatively straightforward to use: https://github.com/google/deepdream/blob/master/dream.ipynb

- gist for osx: https://gist.github.com/robertsdionne/f58a5fc6e5d1d5d2f798

- docker image: https://registry.hub.docker.com/u/mjibson/deepdream/

- booting preinstalled ami + installing caffe at amazon: https://github.com/graphific/dl-machine

- general overview of convolutinal nets using Caffe: https://github.com/graphific/DL-Meetup-intro/blob/master/PyConSe15-cat-vs-dog.ipynb

- or using lasagne: http://www.slideshare.net/roelofp/python-for-image-understanding-deep-learning-with-convolutional-neural-nets


## Credits

Roelof | [KTH](www.csc.kth.se/~roelof/) & [Graph Technologies](http://www.graph-technologies.com/) | [@graphific](https://twitter.com/graphific)
