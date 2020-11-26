#!/bin/bash

#read -p 'image_path: ' img_path
#read -p 'image_out: ' img_out


program='reduction'
img_paths=("4k.png" "720.jpg" "1024.jpg")
out_paths=("4k_480.png" "720_480.jpg" "1024_480.jpg")
size=("4k" "1080p" "720p")

path="../img/"
cmake . && make 

echo "size,block,threads,time" >> stats.txt
for  (( i = 0; i < 3; i++ ));
do
    for (( b = 2; b < 129; b=b*2 ));
    do
        for (( t = 128; t < 8200; t=t*2 ));
        do	
            if [ $(( $t/$b )) -gt 1024 ]; then  
            	break
            fi 
            printf "${b} ${t}\n"
            printf "${size[i]}," >> stats.txt
            ./$program $path${img_paths[i]} $path${out_paths[i]} $t $b >> stats.txt
        done 
    done
done 
