// Copyright (c) 2018 Yi Pan <ypan1988@gmail.com>

#ifndef SAMIN_H_
#define SAMIN_H_

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <R_ext/Print.h>  // Rprintf
#include <R_ext/Random.h> // random number generation in samin()
#include <Rinternals.h>

#include <RcppArmadillo.h>
#include "functor.h"

namespace roptim {
namespace internal {

static double *vect(int n) { return (double *)R_alloc(n, sizeof(double)); }

//template <typename Derived>
inline void genptry(int n, double *p, double *ptry, double scale, void *ex) {
  SEXP s, x;
  int i;
  OptStruct OS = static_cast<Functor *>(ex)->os;
  PROTECT_INDEX ipx;

  if (OS.has_grad_) {
    /* user defined generation of candidate point */
    PROTECT(x = Rf_allocVector(REALSXP, n));
    arma::vec x_copy = arma::zeros<arma::vec>(n);
    for (i = 0; i < n; i++) {
      if (!R_FINITE(p[i])) Rf_error("non-finite value supplied by 'optim'");
      REAL(x)[i] = p[i] * (OS.parscale_(i));
      x_copy(i) = REAL(x)[i];
    }
    arma::vec grad;
    static_cast<Functor *>(ex)->Gradient(x_copy, grad);
    PROTECT_WITH_INDEX(s = Rcpp::wrap(grad), &ipx);
    REPROTECT(s = Rf_coerceVector(s, REALSXP), ipx);
    if (LENGTH(s) != n)
      Rf_error("candidate point in 'optim' evaluated to length %d not %d",
               LENGTH(s), n);
    for (i = 0; i < n; i++) ptry[i] = REAL(s)[i] / (OS.parscale_(i));
    UNPROTECT(2);
  } else { /* default Gaussian Markov kernel */
    for (i = 0; i < n; i++)
      ptry[i] = p[i] + scale * norm_rand(); /* new candidate point */
  }
}

inline
void samin(int n, double *pb, double *yb, optimfn fminfn, int maxit, int tmax,
           double ti, int trace, void *ex)

/* Given a starting point pb[0..n-1], simulated annealing minimization
is performed on the function fminfn. The starting temperature
is input as ti. To make sann work silently set trace to zero.
sann makes in total maxit function evaluations, tmax
evaluations at each temperature. Returned quantities are pb
(the location of the minimum), and yb (the minimum value of
the function func).  Author: Adrian Trapletti
*/
{
  double E1 = 1.7182818; /* exp(1.0)-1.0 */
  double big = 1.0e+35;  /*a very large number*/

  long j;
  int k, its, itdoc;
  double t, y, dy, ytry, scale;
  double *p, *ptry;

  /* Above have: if(trace != 0) trace := REPORT control argument = STEPS */
  if (trace < 0) Rf_error("trace, REPORT must be >= 0 (method = \"SANN\")");

  if (n == 0) { /* don't even attempt to optimize */
    *yb = fminfn(n, pb, ex);
    return;
  }
  p = vect(n);
  ptry = vect(n);
  GetRNGstate();
  *yb = fminfn(n, pb, ex); /* init best system state pb, *yb */
  if (!R_FINITE(*yb)) *yb = big;
  for (j = 0; j < n; j++) p[j] = pb[j];
  y = *yb; /* init system state p, y */
  if (trace) {
    Rprintf("sann objective function values\n");
    Rprintf("initial       value %f\n", *yb);
  }
  scale = 1.0 / ti;
  its = itdoc = 1;
  while (its < maxit) {             /* cool down system */
    t = ti / log((double)its + E1); /* temperature annealing schedule */
    k = 1;
    while ((k <= tmax) && (its < maxit)) /* iterate at constant temperature */
    {
      genptry(n, p, ptry, scale * t,
                       ex); /* generate new candidate point */
      ytry = fminfn(n, ptry, ex);
      if (!R_FINITE(ytry)) ytry = big;
      dy = ytry - y;
      if ((dy <= 0.0) || (unif_rand() < exp(-dy / t))) { /* accept new point? */
        for (j = 0; j < n; j++) p[j] = ptry[j];
        y = ytry;     /* update system state p, y */
        if (y <= *yb) /* if system state is best, then update best system state
                         pb, *yb */
        {
          for (j = 0; j < n; j++) pb[j] = p[j];
          *yb = y;
        }
      }
      its++;
      k++;
    }
    if (trace && ((itdoc % trace) == 0))
      Rprintf("iter %8d value %f\n", its - 1, *yb);
    itdoc++;
  }
  if (trace) {
    Rprintf("final         value %f\n", *yb);
    Rprintf("sann stopped after %d iterations\n", its - 1);
  }
  PutRNGstate();
}

}  // namespace internal
}  // namespace roptim

#endif
