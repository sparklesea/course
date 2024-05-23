# 多节点并行优化
从 NPB 或 SPEC 中选择一个使用 OpenMP 的单节点程序，转换为使用MPI + OpenMP的多节点并行程序

## 要求：

增加MPI通信原语，但不改变openmp计算逻辑

## 提示：

对主要并行区进行并行化，虽然一个应用的源码量有上千行，但主要并行区可能只出现在某几个函数中，需要修改的部分并不多

部分应用虽然可以并行化，但并行化之后通信开销会很大，对于这种情况，请分析通信开销来源。对这些并行区可能存在特殊优化方法，如果能写出特殊优化方法，可以获得bonus。

并行化方法可以参考PPT上的思路： 先进行数据流分析，获得数据模式 -> 再进行任务划分、数据划分 -> 最后增加MPI原语，进行数据分发和回收


参考资料：NPB 官网 (https://www.nas.nasa.gov/software/npb.html) 上有关于 NPB 各个应用的介绍，SPEC 各个应用文件夹下的 Docs 中也有关于应用的描述。

如果同学们使用官网上下载的源码进行修改，记得将编译器修改为 mpicc mpicxx
# NPB
EP 应用已经修改为 MPI 版本，大家可以参考 EP 文件夹下面的代码。

编译时可以选择问题规模，问题规模从小到大包括S A B C D E，建议选择S用于调试，ABC用于实验，也可以自行修改问题规模，使运行时间达到分钟级即可。
## 使用方法
### 以EP为例
cd prj/NPB

make EP CLASS=S

mpirun -n 5 -f netconfig bin/ep.A 10 # process_num 设为 5 threadnum/process设置为 10
# SPEC
官网上的 benchmark 是纯 openmp 的版本，需要修改大量配置文件，不建议从头开始改，这里提供一份修改后的源码，已经进行了MPI初始化，大家可以直接添加 MPI 原语，最后还需要添加 mpi_finalize。

每个应用文件夹下的 data 文件夹包括三种数据文件，ref的数据量非常大，在120并行度下要大概要跑半小时，建议使用train数据集进行调试，实验时自行修改 input 文件大小，使运行时间达到分钟级即可。


## 使用方法：
### 以botsalgn为例：

cd prj/SPEC/358.botsalgn/src

make clean&&make

mpirun -n 4 -f netconfig ./botsalgn -e speccmds.err -o speccmds.stdout -f ../data/ref/input/botsalgn -t 10 # -t参数用于设置一个mpi进程中的omp线程数量，即threadnum/process。该命令将process_num 设为 5，threadnum/process设置为 10

### botsspar 的 input 为 m n，即矩阵大小

mpirun -n 1 -f ~/sycl_convert/node_config/hwconfig6 ./botsspar -m 100 -n 100 -t 10

### kdtree 的 input 为 n cutoffdivisor maxdepth threadnum/process

./kdtree 100000 214748364 2 5 5