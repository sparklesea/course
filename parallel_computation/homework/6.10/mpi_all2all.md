将数据从组收集并分散到组的所有成员。MPI_Alltoall是MPI_Allgather函数的扩展。每个进程都会向每个接收方发送不同的数据。从进程i发送的j块由进程j接收，并放置在接收缓冲区的第i个块中。
int MPIAPI MPI_Alltoall(
  _In_  void         *sendbuf,
        int          sendcount,
        MPI_Datatype sendtype,
  _Out_ void         *recvbuf,
        int          recvcount,
        MPI_Datatype recvtype,
        MPI_Comm     comm
);
sendbuf [in]
指向要发送到组中所有进程的数据的指针。 缓冲区中元素的数量和数据类型在 sendcount 和 sendtype 参数中指定。

如果 comm 参数引用内部通信器，可以通过在所有进程中指定 MPI_IN_PLACE 来指定就地选项。 忽略 sendcount 和 sendtype 参数。 每个进程在相应的接收缓冲区元素中输入数据。 第 n个进程将数据发送到接收缓冲区的 n个元素。

sendcount
在 sendbuf 参数中指定的缓冲区中的元素数。 如果 sendcount 为零，则消息的数据部分为空。

sendtype
发送缓冲区中元素的 MPI 数据类型。

recvbuf [out]
指向包含从每个进程接收的数据的缓冲区的指针。 缓冲区中元素的数量和数据类型在 recvcount 和 recvtype 参数中指定。

recvcount
接收缓冲区中的元素数。 如果计数为零，则消息的数据部分为空。

recvtype
接收缓冲区中元素的 MPI 数据类型。

通讯
MPI_Comm通信器句柄。

MPI_Alltoall
在使用MPI_Alltoall时，每一个进程都会向任意一个进程发送消息，每一个进程也都会接收到任意一个进程的消息。每个进程的接收缓冲区和发送缓冲区都是一个分为若干个数据块的数组。MPI_Alltoall的具体操作是：将进程i的发送缓冲区中的第j块数据发送给进程j，进程j将接收到的来自进程i的数据块放在自身接收缓冲区的第i块位置。
