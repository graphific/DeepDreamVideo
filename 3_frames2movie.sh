#!/bin/bash
if [ $# -eq 0 ]; then
    echo "please provide the directory of the processed frames"
    echo "./2_frames2movie [directory]"
    exit 1
fi

ffmpeg -framerate 25 -i $1/%4d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p tmp.mp4

ffmpeg -i original.mp4 original.mp3
ffmpeg -i original.mp3 music.wav

secs=$(ffprobe -i $1.mp4 -show_entries format=duration -v quiet -of csv="p=0")
ffmpeg -i music.wav -ss 0 -t $secs musicshort.wav
ffmpeg -i musicshort.wav -i tmp.mp4 -strict -2 $1.mp4
rm tmp.mp4
