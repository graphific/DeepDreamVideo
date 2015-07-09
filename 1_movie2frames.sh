#!/bin/bash
if [ $# -ne 3 ]; then
    echo "please provide the moviename and directory where to store the frames"
    echo "usage:"
    echo "./1_movie2frames [ffmpeg|avconv] [movie.ext] [directory]"
    echo "\n example:
    echo "      ./1_movie2frames ffmpeg /home/user/fearloath.mp4 /home/user/framesloath/"
    exit 1
fi

mkdir -p $3
if [ "avconf" == "$1" ]; then
    avconv -i $2 -vsync 1 -an -y -qscale 0 $3/%04d.jpg
else
    ffmpeg -i $2 $3/%4d.jpg
fi
