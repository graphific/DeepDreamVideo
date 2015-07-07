#!/bin/bash
if [ $# -eq 0 ]; then
    echo "please provide the moviename and directory where to store the frames"
    echo "./1_movie2frames [ffmpeg|avconv] [movie.mp4] [directory]"
    exit 1
fi

mkdir -p $3
if [ "avconf" == "$1" ]; then
    avconv -i $2 -vsync 1 -r 25 -an -y -qscale 0 $3/%04d.jpg
else
    ffmpeg -i $2 -r 25.0 $3/%4d.jpg
fi
