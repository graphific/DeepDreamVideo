#!/bin/bash
if [ $# -ne 3 ]; then
    echo "please provide the moviename and directory where to store the frames"
<<<<<<< HEAD
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
=======
    echo "./1_movie2frames [ffmpeg|avconv|mplayer] [movie.mp4] [directory]"
    exit 1
fi

mkdir -p "$3"
rm -R "$3/*"
if [ "avconv" == "$1" ]; then
    AVCONV=$(which avconv)
    FPS=$($AVCONV -i filename 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p") # this line is not tested cuz i don't have avconv :(
    $AVCONV -i "$2" -vsync 1 -r ${FPS} -an -y -qscale 0 "$3/%08d.jpg"
elif [ "mplayer" == "$1" ]; then
    MPLAYER=$(which mplayer)
    # mplayer automatically converts video to images at one image per frame (so no need for the FPS), so the following line is not necessary - but for the record, this is what you would use:
    # FPS=$($MPLAYER -really-quiet -vo null -ao null -frames 0 -identify "${2}" | grep 'ID_VIDEO_FPS' | cut -d'=' -f2)
    $MPLAYER -vo "jpeg:outdir=$3" -ao null "$2"
else
    FFMPEG=$(which ffmpeg)
    FFPROBE=$(which ffprobe)
    FPS=$($FFPROBE -show_streams -select_streams v -i "$2"  2>/dev/null | grep "r_frame_rate" | cut -d'=' -f2)
    $FFMPEG -i "$2" -r ${FPS} -f image2 "$3/%08d.jpg"
>>>>>>> 426b081b064d3748dae053f0efd1892fa7b7100c
fi
