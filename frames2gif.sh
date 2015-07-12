#!/bin/bash
if [ $# -ne 5 ]; then
    echo "please provide the moviename and directory where to store the frames"
    echo "./frames2gif.sh [directory] [movie.mp4] [filename.gif] [gm|im|ffmpeg] [png|jpg]"
    exit 1
fi
    if [ "png" == "$5" ]; then
        suffix="png"
    else
        suffix="jpg"
    fi

    CONVERT=$(which convert)
    GM=$(which gm)
    FFMPEG=$(which ffmpeg)
    FFPROBE=$(which ffprobe)
    FPS=$($FFPROBE -show_streams -select_streams v -i "$2"  2>/dev/null | grep "r_frame_rate" | cut -d'=' -f2 | cut -d'/' -f1)
    echo "FPS: ${FPS}"
if [ "im" == "$4" ]; then # use imagemagick
    FPS=$(echo "1 / ${FPS} * 100" |bc -l)
    $CONVERT "$1/*.${suffix}"  -delay ${FPS} -loop 0 "$3"
elif [ "gm" == "$4" ]; then # use graphicsmagick
    FPS=$(echo "1 / ${FPS} * 100" |bc -l)
    $GM convert "$1/*.${suffix}"  -delay ${FPS} -loop 0 "$3"
else # use crappy gif-algorithm from ffmpeg
    $FFMPEG -f image2 -framerate ${FPS} -i "$1/%08d.${suffix}" "$3"
fi
