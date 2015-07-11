#!/bin/bash
if [ $# -ne 2 ]; then
    echo "please provide the directory of the processed frames"
    echo "usage:"
    echo "./3_frames2movie.sh [frames_directory] [original_video_with_sound]"
    echo "\n example:"
    echo "      ./3_frames2movie.sh /home/user/framesloath/ /home/user/fearloath.mp4"
    exit 1
fi

<<<<<<< HEAD
# FFMPEG NEEDED -- TODO: Develop avconv alternative for this, or delete the avconv option in the first bash script.

# Detect framerate of the original video
frate=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 $2)
ffmpeg -framerate $frate -i $1/%4d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p tmp.mp4 # Use original framerate
=======
FFMPEG=$(which ffmpeg)
FFPROBE=$(which ffprobe)
>>>>>>> 426b081b064d3748dae053f0efd1892fa7b7100c

${FFMPEG} -framerate 25 -i "$1/%08d.jpg" -c:v libx264 -vf "fps=25,format=yuv420p" -tune fastdecode -tune zerolatency -profile:v baseline /tmp/tmp.mp4 -y

<<<<<<< HEAD
# Test if the next part is needed (if framerates are the same, audio should stay in sync)
secs=$(ffprobe -i tmp.mp4 -show_entries format=duration -v quiet -of csv="p=0")
ffmpeg -i music.wav -ss 0 -t $secs musicshort.wav
ffmpeg -i musicshort.wav -i tmp.mp4 -strict -2 $1.mp4
=======
${FFMPEG} -i "$2" /tmp/original.aac -y
#${FFMPEG} -i /tmp/original.mp3 /tmp/music.wav

#secs=$(${FFPROBE} -i /tmp/tmp.mp4 -show_entries format=duration -v quiet -of csv="p=0")
#${FFMPEG} -i /tmp/music.wav -ss 0 -t ${secs} /tmp/musicshort.aac
${FFMPEG} -i /tmp/original.aac -i /tmp/tmp.mp4 -c:v copy -movflags faststart -shortest "${1}_done.mp4" -y
>>>>>>> 426b081b064d3748dae053f0efd1892fa7b7100c

echo 'Removing temp files'
rm /tmp/original.mp3
echo "original.mp3 removed"
rm /tmp/music.aac
echo "music.wav removed"
rm /tmp/musicshort.aac
echo "musicshort.wav removed"
#rm /tmp/tmp.mp4
echo "tmp.mp4 removed"
