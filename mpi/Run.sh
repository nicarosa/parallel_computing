#!/usr/bin/env bash

program='reduccionMPI'
in_path=("4k.jpg" "720p.jpg" "1080p.jpg")
out_path=("4k_480.jpg" "720p_480.jpg" "1080p_480.jpg")
size=("4k" "720p" "1080p")

path="../images/"
cmake . && make 

echo "Tipo,Hilos Usados,Tiempo de ejecucion" >> resultados.txt
for  (( i = 0; i < 3; i++ ));
do 
    for t in 2 4;
    do
        printf "${size[i]}," >> resultados.txt
        mpirun -np $t --use-hwthread-cpus $program $path${in_path[i]} $path${out_path[i]}  >> resultados.txt
    done
done 
