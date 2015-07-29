#!/bin/bash
if [ $# -ne 4 ]; then
    echo "please provide the moviename and directory where to store the frames"
    echo "./1_movie2frames [ffmpeg|avconv|mplayer] [movie.mp4] [directory] [png|jpg]"
    exit 1
fi

mkdir -p "$3"
echo "Removing files in $3/*"
rm -R "$3/"*

if [ "png" == "$4" ]; then
    OUTFILES="$3/%08d.png"
    MPLAYERCOMMAND="png:z=9:outdir=$3"
else
    OUTFILES="$3/%08d.jpg"
    MPLAYERCOMMAND="jpg:outdir=$3"
fi

if [ "avconv" == "$1" ]; then
    AVCONV=$(which avconv)
    FPS=$($AVCONV -i "$2" 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")
    $AVCONV -i "$2" -vsync 1 -r ${FPS} -an -y -qscale 0 "${OUTFILES}"
elif [ "mplayer" == "$1" ]; then
    MPLAYER=$(which mplayer)
    # mplayer automatically converts video to images at one image per frame (so no need for the FPS), so the following line is not necessary - but for the record, this is what you would use:
    # FPS=$($MPLAYER -really-quiet -vo null -ao null -frames 0 -identify "${2}" | grep 'ID_VIDEO_FPS' | cut -d'=' -f2)
    $MPLAYER -vo "${MPLAYERCOMMAND}" -ao null "$2"
else
    FFMPEG=$(which ffmpeg)
    #Same happens with FFMPEG as MPLAYER, fps is not needed for video to frame convertion.
    #FFPROBE=$(which ffprobe)
    #FPS=$($FFPROBE -show_streams -select_streams v -i "$2" 2>/dev/null | grep "r_frame_rate" | cut -d'=' -f2)
    "$FFMPEG" -i "$2" -f image2 "${OUTFILES}"
fi

if [ "png" == "$4" ]; then
    PNGCRUSH=$(which pngcrush)
    if [ "${PNGCRUSH}" != "" ]; then
        for f in $(find "$3" -type f); do
            # using method 115 because on my test material it worked the best
            # if you really have a lot of time on your hands you could use the
            # second version commented out
            echo "PNGCRUSHING: $f"
            ${PNGCRUSH} -ow -m 115 "$f" >/dev/null 2>&1
            #${PNGCRUSH} -ow -brute "$f" >/dev/null 2>&1
        done
    else
        echo "pngcrush not installed, can't crush the images"
    fi
fi
