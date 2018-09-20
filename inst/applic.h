// Copyright (c) 2018 Yi Pan <ypan1988@gmail.com>
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available at
//  https://www.R-project.org/Licenses/

#ifndef APPLIC_H_
#define APPLIC_H_

// This file is part of R_ext/Applic.h for the function prototype of
// optimization routines. I am not able to include R_ext/Applic.h in roptim.h
// directly since it conflict with RcppArmadillo.h.

#ifdef __cplusplus
extern "C" {
#endif

typedef double optimfn(int, double *, void *);
typedef void optimgr(int, double *, double *, void *);

void vmmin(int n, double *b, double *Fmin, optimfn fn, optimgr gr, int maxit,
           int trace, int *mask, double abstol, double reltol, int nREPORT,
           void *ex, int *fncount, int *grcount, int *fail);
void nmmin(int n, double *Bvec, double *X, double *Fmin, optimfn fn, int *fail,
           double abstol, double intol, void *ex, double alpha, double bet,
           double gamm, int trace, int *fncount, int maxit);
void cgmin(int n, double *Bvec, double *X, double *Fmin, optimfn fn, optimgr gr,
           int *fail, double abstol, double intol, void *ex, int type,
           int trace, int *fncount, int *grcount, int maxit);
void lbfgsb(int n, int m, double *x, double *l, double *u, int *nbd,
            double *Fmin, optimfn fn, optimgr gr, int *fail, void *ex,
            double factr, double pgtol, int *fncount, int *grcount, int maxit,
            char *msg, int trace, int nREPORT);
void samin(int n, double *pb, double *yb, optimfn fn, int maxit, int tmax,
           double ti, int trace, void *ex);

#ifdef __cplusplus
}
#endif

#endif  // APPLIC_H_
