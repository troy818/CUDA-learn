#ifndef GPU_H_
#define GPU_H_

void NaiveErosionOneStep(int * src, int * dst, int width, int height, int radio);
void NaiveErosionOneStepMod(int * src, int * dst, int width, int height, int radio);

void ErosionTwoSteps(int * src, int * dst, int * temp, int width, int height, int radio);
void ErosionTwoStepsShared(int * src, int * dst, int * temp, int width, int height, int radio);

#endif  // GPU_H_