#!/bin/bash
if [ $# -ne 2 ]; then
    echo "please provide the directory of the processed frames"
    echo "usage:"
    echo "./3_frames2movie.sh [frames_directory] [original_video_with_sound]"
    echo "\n example:"
    echo "      ./3_frames2movie.sh /home/user/framesloath/ /home/user/fearloath.mp4"
    exit 1
fi

# FFMPEG NEEDED -- TODO: Develop avconv alternative for this, or delete the avconv option in the first bash script.

# Detect framerate of the original video
frate=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 $2)
ffmpeg -framerate $frate -i $1/%4d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p tmp.mp4 # Use original framerate

ffmpeg -i $2 original.mp3
ffmpeg -i original.mp3 music.wav

# Test if the next part is needed (if framerates are the same, audio should stay in sync)
secs=$(ffprobe -i tmp.mp4 -show_entries format=duration -v quiet -of csv="p=0")
ffmpeg -i music.wav -ss 0 -t $secs musicshort.wav
ffmpeg -i musicshort.wav -i tmp.mp4 -strict -2 $1.mp4

echo 'Removing temp files'
rm original.mp3
echo "original.mp3 removed"
rm music.wav
echo "music.wav removed"
rm musicshort.wav
echo "musicshort.wav removed"
rm tmp.mp4
echo "tmp.mp4 removed"
