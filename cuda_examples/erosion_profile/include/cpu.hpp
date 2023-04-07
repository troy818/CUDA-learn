#ifndef CPU_H_
#define CPU_H_

void erosionCPUOneStep(int * src, int * dst, int width, int height, int radio);
void erosionCPUTwoStep(int * src, int * dst, int width, int height, int radio);

#endif  // CPU_H_