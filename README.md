roptim: General Purpose Optimization in R using C++.
====

[![Build Status](https://travis-ci.org/ypan1988/roptim.svg?branch=master)](https://travis-ci.org/ypan1988/roptim)
[![cran version](http://www.r-pkg.org/badges/version/roptim)](https://cran.r-project.org/web/packages/roptim)
[![downloads](http://cranlogs.r-pkg.org/badges/roptim)](http://cranlogs.r-pkg.org/badges/roptim)
[![total downloads](http://cranlogs.r-pkg.org/badges/grand-total/roptim)](http://cranlogs.r-pkg.org/badges/grand-total/roptim)

## Installation

Get the development version from github:
```R
install.packages("devtools")
library(devtools)
devtools::install_github("ypan1988/roptim", dependencies=TRUE)
```

Or the released version from CRAN:
```R
install.packages("roptim")
```

## A Quick Example
An example of using `stats::optim()` in `R` environment:
```R
fr <- function(x) {   ## Rosenbrock Banana function
    x1 <- x[1]
    x2 <- x[2]
    100 * (x2 - x1 * x1)^2 + (1 - x1)^2
}
grr <- function(x) { ## Gradient of 'fr'
    x1 <- x[1]
    x2 <- x[2]
    c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
       200 *      (x2 - x1 * x1))
}
res <- optim(c(-1.2,1), fr, grr, method = "BFGS", control = list(trace=T), hessian = TRUE)

```

Corresponding code written in `C++` using package `roptim` (file `demo.cpp`):
```cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <roptim.h>
// [[Rcpp::depends(roptim)]]

using namespace roptim;

class Rosen : public Functor {
public:
  double operator()(const arma::vec &x) override {
    double x1 = x(0);
    double x2 = x(1);
    return 100 * std::pow((x2 - x1 * x1), 2) + std::pow(1 - x1, 2);
  }
  
  void Gradient(const arma::vec &x, arma::vec &gr) override {
    gr = arma::zeros<arma::vec>(2);
    
    double x1 = x(0);
    double x2 = x(1);
    gr(0) = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
    gr(1) = 200 * (x2 - x1 * x1);
  }
};


// [[Rcpp::export]]
void rosen_bfgs() {
  Rosen rb;
  Roptim<Rosen> opt("BFGS");
  opt.control.trace = 1;
  opt.set_hessian(true);
  
  arma::vec x = {-1.2, 1};
  opt.minimize(rb, x);
  
  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();
}
```

Compile and run the function in `R`:
```R
sourceCpp("~/demo.cpp") # you may need to change the directory
rosen_bfgs()
```

Then you will get expected output as follows:
```
initial  value 24.200000 
iter  10 value 1.367383
iter  20 value 0.134560
iter  30 value 0.001978
iter  40 value 0.000000
final  value 0.000000 
converged
-------------------------
.par()
   1.0000   1.0000

.value()
9.59496e-018

.fncount()
110

.grcount()
43

.convergence()
0

.message()
NULL

.hessian()
  8.0200e+002 -4.0000e+002
 -4.0000e+002  2.0000e+002
```