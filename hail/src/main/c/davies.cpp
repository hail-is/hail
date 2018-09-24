#include "davies.h"

#define EXPORT __attribute__((visibility("default")))

DaviesAlgo::DaviesAlgo(){}

real DaviesAlgo::exp1(real x){ return x < -50.0 ? 0.0 : exp(x); }

void DaviesAlgo::counter(void)
{
  count = count + 1;
  if ( count > lim ) longjmp(env,1);
}

real DaviesAlgo::square(real x) {return x*x;}
real DaviesAlgo::cube(real x) {return x*x*x;}

real DaviesAlgo::log1(real x,BOOL first)
{
  if (fabs(x) > 0.1)
    {
      return (first ? log(1.0 + x) : (log(1.0 + x) - x));
    }
  else
    {
      real s, s1, term, y, k;
      y = x / (2.0 + x);  term = 2.0 * cube(y);  k = 3.0;
      s = (first ? 2.0 : - x) * y;
      y = square(y);
      for (s1 = s + term / k; s1 != s; s1 = s + term / k)
	{k = k + 2.0; term = term * y; s = s1; }
      return s;
    }
}

void DaviesAlgo::order()
{
  int j, k; real lj;
  
  for ( j=0; j<r; j++ )
    {
      lj = fabs(lb[j]);
      for (k = j-1; k>=0; k--)
	{
	  if ( lj > fabs(lb[th[k]]) )  th[k + 1] = th[k];
	  else goto l1;
	}
      k = -1;
    l1 :
      th[k + 1] = j;
    }
  ndtsrt = FALSE;
}

real DaviesAlgo::errbd(real u, real* cx)
{
  real sum1, lj, ncj, x, y, xconst; int j, nj;
  
  counter();
  xconst = u * sigsq;  sum1 = u * xconst;  u = 2.0 * u;
  for (j=r-1; j>=0; j--)
    {
      nj = n[j]; lj = lb[j]; ncj = nc[j];
      x = u * lj; y = 1.0 - x;
      xconst = xconst + lj * (ncj / y + nj) / y;
      sum1 = sum1 + ncj * square(x / y)
	+ nj * (square(x) / y + log1(-x, FALSE ));
    }
  *cx = xconst; return exp1(-0.5 * sum1);
}

real DaviesAlgo::ctff(real accx, real* upn)
{
  real u1, u2, u, rb, xconst, c1, c2;
  
  u2 = *upn;   u1 = 0.0;  c1 = mean;
  rb = 2.0 * ((u2 > 0.0) ? lmax : lmin);
  for (u = u2 / (1.0 + u2 * rb); errbd(u, &c2) > accx;
       u = u2 / (1.0 + u2 * rb))
    {
      u1 = u2;  c1 = c2;  u2 = 2.0 * u2;
    }
  for (u = (c1 - mean) / (c2 - mean); u < 0.9;
       u = (c1 - mean) / (c2 - mean))
    {
      u = (u1 + u2) / 2.0;
      if (errbd(u / (1.0 + u * rb), &xconst) > accx)
	{  u1 = u; c1 = xconst;  }
      else
	{  u2 = u;  c2 = xconst; }
    }
  *upn = u2; return c2;
}

real DaviesAlgo::truncation(real u,real tausq)
{
  real sum1, sum2, prod1, prod2, prod3, lj, ncj,
    x, y, err1, err2;
  int j, nj, s;

  counter();
  sum1  = 0.0; prod2 = 0.0;  prod3 = 0.0;  s = 0;
  sum2 = (sigsq + tausq) * square(u); prod1 = 2.0 * sum2;
  u = 2.0 * u;
  for (j=0; j<r; j++ )
    {
      lj = lb[j];  ncj = nc[j]; nj = n[j];
      x = square(u * lj);
      sum1 = sum1 + ncj * x / (1.0 + x);
      if (x > 1.0)
	{
	  prod2 = prod2 + nj * log(x);
	  prod3 = prod3 + nj * log1(x, TRUE );
	  s = s + nj;
	}
      else  prod1 = prod1 + nj * log1(x, TRUE );
    }
  sum1 = 0.5 * sum1;
  prod2 = prod1 + prod2;  prod3 = prod1 + prod3;
  x = exp1(-sum1 - 0.25 * prod2) / pi;
  y = exp1(-sum1 - 0.25 * prod3) / pi;
  err1 =  ( s  ==  0 )  ? 1.0 : x * 2.0 / s;
  err2 =  ( prod3 > 1.0 )  ? 2.5 * y : 1.0;
  if (err2 < err1) err1 = err2;
  x = 0.5 * sum2;
  err2 =  ( x  <=  y )  ? 1.0  : y / x;
  return  ( err1 < err2 )  ? err1  :  err2;
}

void DaviesAlgo::findu(real* utx, real accx)
{
  real u, ut; int i;
  static real divis[]={2.0,1.4,1.2,1.1};
  ut = *utx; u = ut / 4.0;
  if ( truncation(u, 0.0) > accx )
    {
      for ( u = ut; truncation(u, 0.0) > accx; u = ut) ut = ut * 4.0;
    }
  else
    {
      ut = u;
      for ( u = u / 4.0; truncation(u, 0.0) <=  accx; u = u / 4.0 )
	ut = u;
    }
  for ( i=0;i<4;i++)
    { u = ut/divis[i]; if ( truncation(u, 0.0)  <=  accx )  ut = u; }
  *utx = ut;
}

void DaviesAlgo::integrate(int nterm, real interv,real tausq,BOOL mainx)
{
  real inpi, u, sum1, sum2, sum3, x, y, z;
  int k, j, nj;
 
  inpi = interv / pi;
  for ( k = nterm; k>=0; k--)
    {
      u = (k + 0.5) * interv;
      sum1 = - 2.0 * u * c;  sum2 = fabs(sum1);
      sum3 = - 0.5 * sigsq * square(u);
      for ( j = r-1; j>=0; j--)
	{
	  nj = n[j];  x = 2.0 * lb[j] * u;  y = square(x);
	  sum3 = sum3 - 0.25 * nj * log1(y, TRUE );
	  y = nc[j] * x / (1.0 + y);
	  z = nj * atan(x) + y;
	  sum1 = sum1 + z;   sum2 = sum2 + fabs(z);
	  sum3 = sum3 - 0.5 * x * y;
	}
      x = inpi * exp1(sum3) / u;
      if ( !  mainx )
	x = x * (1.0 - exp1(-0.5 * tausq * square(u)));
      sum1 = sin(0.5 * sum1) * x;  sum2 = 0.5 * sum2 * x;
      intl = intl + sum1; ersm = ersm + sum2;
    }
}

real DaviesAlgo::cfe(real x)
{
  real axl, axl1, axl2, sxl, sum1, lj; int j, k, t;

  counter();
  if (ndtsrt) order();
  axl = fabs(x);  sxl = (x>0.0) ? 1.0 : -1.0;  sum1 = 0.0;
  for ( j = r-1; j>=0; j-- )
    { t = th[j];
      if ( lb[t] * sxl > 0.0 )
	{
	  lj = fabs(lb[t]);
	  axl1 = axl - lj * (n[t] + nc[t]);  axl2 = lj / log28;
	  if ( axl1 > axl2 )  axl = axl1  ; else
            {
	      if ( axl > axl2 )  axl = axl2;
	      sum1 = (axl - axl1) / lj;
	      for ( k = j-1; k>=0; k--)
		sum1 = sum1 + (n[th[k]] + nc[th[k]]);
	      goto  l;
            }
	}
    }
 l:
  if (sum1 > 100.0)
    { fail = TRUE; return 1.0; } else
    return pow(2.0,(sum1 / 4.0)) / (pi * square(axl));
}


real DaviesAlgo::qf(real* lb1, real* nc1, int* n1, int r1, real sigma, real c1,
		    int lim1, real acc, real* trace, int* ifault)

/*  distribution function of a linear combination of non-central                    chi-squared random variables :                                                
                                                                                 
input:                                                                           
   lb[j]            coefficient of j-th chi-squared variable                    
   nc[j]            non-centrality parameter                                    
   n[j]             degrees of freedom                                          
   j = 0, 2 ... r-1                                                             
   sigma            coefficient of standard normal variable                     
   c                point at which df is to be evaluated                        
   lim              maximum number of terms in integration                      
   acc              maximum error                                               
                                                                                
output:                                                                          
   ifault = 1       required accuracy NOT achieved                              
            2       round-off error possibly significant                        
            3       invalid parameters                                          
            4       unable to locate integration parameters                     
            5       out of memory
   
   trace[0]         absolute sum                                                
   trace[1]         total number of integration terms                           
   trace[2]         number of integrations                                      
   trace[3]         integration interval in final integration                   
   trace[4]         truncation point in initial integration                     
   trace[5]         s.d. of initial convergence factor                          
   trace[6]         cycles to locate integration parameters     */

{
  int j, nj, nt, ntm;  real acc1, almx, xlim, xnt, xntm;
  real utx, tausq, sd, intv, intv1, x, up, un, d1, d2, lj, ncj;
  real qfval = -1.0;
  static int rats[]={1,2,4,8};
  
  if (setjmp(env) != 0) { *ifault=4; goto endofproc; }
  r=r1; lim=lim1; c=c1;
  n=n1; lb=lb1; nc=nc1;
  for ( j = 0; j<7; j++ )  trace[j] = 0.0;
  *ifault = 0; count = 0;
  intl = 0.0; ersm = 0.0;
  qfval = -1.0; acc1 = acc; ndtsrt = TRUE;  fail = FALSE;
  xlim = (real)lim;
  th=(int*)malloc(r*(sizeof(int)));
  if (! th) { *ifault=5;  goto  endofproc; }

  /* find mean, sd, max and min of lb,                                       
     check that parameter values are valid */
  sigsq = square(sigma); sd = sigsq;
  lmax = 0.0; lmin = 0.0; mean = 0.0;
  for (j=0; j<r; j++ )
    {
      nj = n[j];  lj = lb[j];  ncj = nc[j];
      if ( nj < 0  ||  ncj < 0.0 ) { *ifault = 3;  goto  endofproc;  }
      sd  = sd  + square(lj) * (2 * nj + 4.0 * ncj);
      mean = mean + lj * (nj + ncj);
      if (lmax < lj) lmax = lj ; else if (lmin > lj) lmin = lj;
    }
  if ( sd == 0.0  )
    {  qfval = (c > 0.0) ? 1.0 : 0.0; goto  endofproc;  }
  if ( lmin == 0.0 && lmax == 0.0 && sigma == 0.0 )
    { *ifault = 3;  goto  endofproc;  }
  sd = sqrt(sd);
  almx = (lmax < - lmin) ? - lmin : lmax;

  /* starting values for findu, ctff */
  utx = 16.0 / sd;  up = 4.5 / sd;  un = - up;
  /* truncation point with no convergence factor */
  findu(&utx, .5 * acc1);
  /* does convergence factor help */
  if (c != 0.0  && (almx > 0.07 * sd))
    {
      tausq = .25 * acc1 / cfe(c);
      if (fail) fail = FALSE ;
      else if (truncation(utx, tausq) < .2 * acc1)
	{
	  sigsq = sigsq + tausq;
	  findu(&utx, .25 * acc1);
	  trace[5] = sqrt(tausq);
	}
    }
  trace[4] = utx;  acc1 = 0.5 * acc1;

  /* find RANGE of distribution, quit if outside this */
 l1:
  d1 = ctff(acc1, &up) - c;
  if (d1 < 0.0) { qfval = 1.0; goto endofproc; }
  d2 = c - ctff(acc1, &un);
  if (d2 < 0.0) { qfval = 0.0; goto endofproc; }
  /* find integration interval */
  intv = 2.0 * pi / ((d1 > d2) ? d1 : d2);
  /* calculate number of terms required for main and                         
     auxillary integrations */
  xnt = utx / intv;  xntm = 3.0 / sqrt(acc1);
  if (xnt > xntm * 1.5)
  {
     /* parameters for auxillary integration */
     if (xntm > xlim) { *ifault = 1; goto endofproc; }
     ntm = (int)floor(xntm+0.5);
     intv1 = utx / ntm;  x = 2.0 * pi / intv1;
     if (x <= fabs(c)) goto l2;
     /* calculate convergence factor */
     tausq = .33 * acc1 / (1.1 * (cfe(c - x) + cfe(c + x)));
     if (fail) goto l2;
     acc1 = .67 * acc1;
     /* auxillary integration */
     integrate(ntm, intv1, tausq, FALSE );
     xlim = xlim - xntm;  sigsq = sigsq + tausq;
     trace[2] = trace[2] + 1; trace[1] = trace[1] + ntm + 1;
     /* find truncation point with new convergence factor */
     findu(&utx, .25 * acc1);  acc1 = 0.75 * acc1;
     goto l1;
   }
 
   /* main integration */
 l2:

  trace[3] = intv;
  if (xnt > xlim) { *ifault = 1; goto endofproc; }
  nt = (int)floor(xnt+0.5);
  integrate(nt, intv, 0.0, TRUE );
  trace[2] = trace[2] + 1; trace[1] = trace[1] + nt + 1;
  qfval = 0.5 - intl;
  trace[0] = ersm;

  /* test whether round-off error could be significant                       
     allow for radix 8 or 16 machines */
  up=ersm; x = up + acc / 10.0;
  for (j=0;j<4;j++) { if (rats[j] * x == rats[j] * up) *ifault = 2; }

 endofproc :
  free((char*)th);
  trace[6] = (real)count;
  return qfval;
}

extern "C"
EXPORT
real qfWrapper(real* lb1, real* nc1, int* n1, int r1, real sigma, real c1,
               int lim1, real acc, real* trace, int* ifault)
{

  real lambdas[r1];
  real noncentrality[r1];
  int dof[r1];

  for(int i = 0; i < r1; i++){
    lambdas[i] = lb1[i];
    noncentrality[i] = nc1[i];
    dof[i] = 1;
  }

  real trace2[7] = {0.0};
  int fault = *ifault;

  DaviesAlgo algoClass;
 
  real result = algoClass.qf(lambdas, noncentrality, dof, r1, sigma, c1,
                   lim1, acc, trace2, &fault);
  *ifault = fault;
  return result;

}

