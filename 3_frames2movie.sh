#!/bin/bash
if [ $# -ne 2 ]; then
    echo "please provide the directory of the processed frames"
    echo "./3_frames2movie.sh [frames_directory] [original_video_with_sound]"
    exit 1
fi

FFMPEG=$(which ffmpeg)
FFPROBE=$(which ffprobe)

${FFMPEG} -framerate 25 -i "$1/%08d.jpg" -c:v libx264 -vf "fps=25,format=yuv420p" -tune fastdecode -tune zerolatency -profile:v baseline /tmp/tmp.mp4 -y

${FFMPEG} -i "$2" /tmp/original.aac -y
#${FFMPEG} -i /tmp/original.mp3 /tmp/music.wav

#secs=$(${FFPROBE} -i /tmp/tmp.mp4 -show_entries format=duration -v quiet -of csv="p=0")
#${FFMPEG} -i /tmp/music.wav -ss 0 -t ${secs} /tmp/musicshort.aac
${FFMPEG} -i /tmp/original.aac -i /tmp/tmp.mp4 -c:v copy -movflags faststart -shortest "${1}TEST.mp4" -y

echo 'Removing temp files'
rm /tmp/original.mp3
echo "original.mp3 removed"
rm /tmp/music.aac
echo "music.wav removed"
rm /tmp/musicshort.aac
echo "musicshort.wav removed"
#rm /tmp/tmp.mp4
echo "tmp.mp4 removed"
