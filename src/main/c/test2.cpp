#include "ibs.h"

#define EXPORT __attribute__((visibility("default")))

typedef double real;

extern "C"
EXPORT
real qf(real*,real*,int*,int,real,real,int,real,real*,int*);


int main(int argc, char ** argv) {
  
  real lambdas[3] = {6.0,3.0,1.0};
  real noncentrality[3] = {0,0,0};
  int degreesOfFreedom[3] = {1,1,1};
  real s = 0.0;
  int terms = 3;
  real Q = 1.0;
  real trace[7] = {0,0,0,0,0,0,0};
  int response = 0;
    qf(lambdas,noncentrality,degreesOfFreedom,terms,s
       ,Q,1000,0.0001,trace,&response);

  return 0;
}

