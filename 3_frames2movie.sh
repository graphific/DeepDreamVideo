#!/bin/bash
if [ $# -ne 4 ]; then
    echo "please provide the directory of the processed frames"
    echo "./3_frames2movie.sh [ffmpeg|avconv|mplayer] [frames_directory] [original_video_with_sound] [png|jpg]"
    exit 1
fi

if [ "png" == "$4" ]; then
    INFILES="$2/%08d.png"
    MENCODERCOMMAND="type=png"
else
    INFILES="$2/%08d.jpg"
    MENCODERCOMMAND="type=jpg"
fi

CODEC="libx264"
OUTFILE="$(basename $2)_done.mp4"
TMPAUDIO="/tmp/tmp.aac"
TMPVIDEO="/tmp/tmp.mp4"

if [ -f ${OUTFILE} ]; then
    rm "${OUTFILE}"
fi

if [ "avconv" == "$1" ]; then
    AVCONV=$(which avconv)
    BITRATE=$($AVCONV -i "$3" 2>&1 | sed -n "s/.*, \([0-9].*\) kb\/s.*/\1/p")
    FPS=$($AVCONV -i "$3" 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")

    "${AVCONV}" -r "${FPS}" -i "${INFILES}" -b:v "${BITRATE}k" -c:v "${CODEC}" -vf "format=yuv420p" "${TMPVIDEO}" -y
    "${AVCONV}" -i "$3" -strict -2 "${TMPAUDIO}" -y

    if [ -s "${TMPAUDIO}" ]; then
        "${AVCONV}" -i "${TMPAUDIO}" -i "${TMPVIDEO}" -strict -2 -c:v copy -shortest "${OUTFILE}"
    else
        "${AVCONV}"                  -i "${TMPVIDEO}" -strict -2 -c:v copy -shortest "${OUTFILE}"
    fi
elif [ "mplayer" == "$1" ]; then
    MENCODER=$(which mencoder)
    MPLAYER=$(which mplayer)
    BITRATE=$(( $($MPLAYER -really-quiet -vo null -ao null -frames 0 -identify "${3}" | grep 'ID_VIDEO_BITRATE' | cut -d'=' -f2) / 1000 ))
    FPS=$($MPLAYER -really-quiet -vo null -ao null -frames 0 -identify "${3}" | grep 'ID_VIDEO_FPS' | cut -d'=' -f2)

    "${MENCODER}" "mf://${INFILES}" -mf "fps=${FPS}:${MENCODERCOMMAND}" -oac copy -ovc x264 -x264encopts bitrate=${BITRATE} -ofps ${FPS} -o "${OUTFILE}"
    # Missing audio
else
    FFMPEG=$(which ffmpeg)
    FFPROBE=$(which ffprobe)
    FPS=$(${FFPROBE} -show_streams -select_streams v -i "$3" 2>/dev/null | grep "r_frame_rate" | cut -d'=' -f2)

    "${FFMPEG}" -framerate ${FPS} -i "${INFILES}" -c:v ${CODEC} -vf "fps=${FPS},format=yuv420p" -tune fastdecode -tune zerolatency -profile:v baseline "${TMPVIDEO}" -y

    "${FFMPEG}" -i "$3" -strict -2 "${TMPAUDIO}" -y
    "${FFMPEG}" -i "${TMPAUDIO}" /tmp/music.wav

    #secs=$(${FFPROBE} -i "${TMPVIDEO}" -show_entries format=duration -v quiet -of csv="p=0")
    #${FFMPEG} -i /tmp/music.wav -ss 0 -t ${secs} /tmp/musicshort.aac
    "${FFMPEG}" -i "${TMPAUDIO}" -i "${TMPVIDEO}" -strict -2 -c:v copy -movflags faststart -shortest "${OUTFILE}"
fi

echo "Removing temp files"
rm "${TMPAUDIO}"
#echo "${TMPAUDIO} removed"
#rm /tmp/music.aac
#echo "/tmp/music.wav removed"
#rm /tmp/musicshort.aac
#echo "/tmp/musicshort.wav removed"
rm "${TMPVIDEO}"
#echo "${TMPVIDEO} removed"

if [ -s ${OUTFILE} ]; then
    echo "saved movie as: ${OUTFILE}"
fi
