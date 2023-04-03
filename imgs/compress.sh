#!/bin/sh

palette="/tmp/palette.png"

filters="fps=15,scale=450:-1:flags=lanczos"

ffmpeg -v warning -i $1 -vf "$filters,palettegen=stats_mode=diff" -y $palette

ffmpeg -i $1 -i $palette -lavfi "$filters,paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y $2
