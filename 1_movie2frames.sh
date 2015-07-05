!#/bin/bash
if [ $# -eq 0 ]; then
    echo "please provide the moviename and directory where to store the frames"
    echo "./1_movie2frames [movie.mp4] [directory]"
    exit 1
fi

ffmpeg -i $1 -r 25.0 $2/%4d.jpg
