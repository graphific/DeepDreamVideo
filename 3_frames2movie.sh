#!/bin/bash
if [ $# -ne 3 ]; then
    echo "please provide the directory of the processed frames"
    echo "./3_frames2movie.sh [frames_directory] [original_video_with_sound] [png|jpg]"
    exit 1
fi

if [ "png" == "$3" ]; then
    suffix="png"
else
    suffix="jpg"
fi

FFMPEG=$(which ffmpeg)
FFPROBE=$(which ffprobe)

FPS=$(${FFPROBE} -show_streams -select_streams v -i "$2"  2>/dev/null | grep "r_frame_rate" | cut -d'=' -f2)


${FFMPEG} -framerate ${FPS} -i "$1/%08d.${suffix}" -c:v libx264 -vf "fps=${FPS},format=yuv420p" -tune fastdecode -tune zerolatency -profile:v baseline /tmp/tmp.mp4 -y

${FFMPEG} -i "$2" -strict -2 /tmp/original.aac -y
#${FFMPEG} -i /tmp/original.aac /tmp/music.wav

#secs=$(${FFPROBE} -i /tmp/tmp.mp4 -show_entries format=duration -v quiet -of csv="p=0")
#${FFMPEG} -i /tmp/music.wav -ss 0 -t ${secs} /tmp/musicshort.aac
${FFMPEG} -i /tmp/original.aac -i /tmp/tmp.mp4 -strict -2 -c:v copy -movflags faststart -shortest "${1}_done.mp4" -y

echo 'Removing temp files'
rm /tmp/original.aac
#echo "original.aac removed"
#rm /tmp/music.aac
#echo "music.wav removed"
#rm /tmp/musicshort.aac
#echo "musicshort.wav removed"
rm /tmp/tmp.mp4
#echo "tmp.mp4 removed"

echo "saved movie as: ${1}_done.mp4"
