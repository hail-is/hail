#ifndef DAVIES_HEADER
#define DAVIES_HEADER

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<setjmp.h>

typedef double real;
#define TRUE 1
#define FALSE 0

#define pi 3.14159265358979
#define log28 .0866

typedef int BOOL;

class DaviesAlgo {
 private:
  real sigsq,lmax,lmin,mean,c;
  double intl, ersm;
  int count, r, lim;
  BOOL ndtsrt, fail;
  int *n, *th;
  real *lb, *nc;
  jmp_buf env;
 public:
  DaviesAlgo();
  real exp1(real x);
  void counter(void);
  real square(real);
  real cube(real);
  real log1(real,BOOL);
  void order(void);
  real errbd(real u, real* cx);
  real ctff(real,real*);
  real truncation(real,real);
  void findu(real*,real);
  void integrate(int,real,real,BOOL);
  real cfe(real);
  real qf(real*,real*,int*,int,real,real,int,real,real*,int*);
};


#endif
