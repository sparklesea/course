#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
 
void MatMul(int *M,int *N,int *P,int width);

 
int main()
{
	const int m = 30;
	int a[m][m],b[m][m],c[m][m];	
	int width = m;

	for(int i = 0;i < m;i++)
	{
		for(int j = 0;j < m;j++)
		{
			a[i][j] = 10;
			b[i][j] = 20;
		}
	}
	
    MatMul(a, b, c, width);
	
	return 0;
}