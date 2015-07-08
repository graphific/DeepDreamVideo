#!/bin/bash
if [ $# -eq 0 ]; then
    echo "please provide the moviename and directory where to store the frames"
    echo "./frames2gif.sh [directory] [frames_a_second] [filename.gif]"
    exit 1
fi

ffmpeg -f image2 -framerate $2 -i $1/%04d.jpg $3
