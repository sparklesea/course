make MG CLASS=$1

mpirun -n $2 bin/mg.$1 $3