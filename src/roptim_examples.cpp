#include <cmath>

#include <algorithm>

#include <RcppArmadillo.h>
#include "roptim.h"

using namespace roptim;

//////////////////////////////////////////////////
// EXAMPLE 1
//////////////////////////////////////////////////

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

  void Hessian(const arma::vec &x, arma::mat &he) override {
    he = arma::zeros<arma::mat>(2, 2);

    double x1 = x(0);
    double x2 = x(1);
    he(0, 0) = -400 * x2 + 1200 * x1 * x1 + 2;
    he(0, 1) = -400 * x1;
    he(1, 0) = he(0, 1);
    he(1, 1) = 200;
  }
};

class RosenNoGrad : public Functor {
 public:
  double operator()(const arma::vec &x) override {
    double x1 = x(0);
    double x2 = x(1);
    return 100 * std::pow((x2 - x1 * x1), 2) + std::pow(1 - x1, 2);
  }
};

//'@title Example 1: Minimize Rosenbrock function using BFGS
//'@description Minimize Rosenbrock function using BFGS.
//'@examples
//'fr <- function(x) {   ## Rosenbrock Banana function
//'  x1 <- x[1]
//'  x2 <- x[2]
//'  100 * (x2 - x1 * x1)^2 + (1 - x1)^2
//'}
//'grr <- function(x) { ## Gradient of 'fr'
//'  x1 <- x[1]
//'  x2 <- x[2]
//'  c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
//'    200 *      (x2 - x1 * x1))
//'}
//'res <- optim(c(-1.2,1), fr, grr, method = "BFGS", control = list(trace=TRUE), hessian = TRUE)
//'res
//'
//'## corresponding C++ implementation:
//'example1_rosen_bfgs()
//'@export
// [[Rcpp::export]]
void example1_rosen_bfgs() {
  Rosen rb;
  Roptim<Rosen> opt("BFGS");
  opt.control.trace = 1;
  opt.set_hessian(true);

  arma::vec x = {-1.2, 1};
  opt.minimize(rb, x);

  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();
}

//'@title Example 1: Minimize Rosenbrock function using other methods
//'@description Minimize Rosenbrock function using other methods ("Nelder-Mead"/"CG"/ "L-BFGS-B"/"SANN").
//'@examples
//'fr <- function(x) {   ## Rosenbrock Banana function
//'  x1 <- x[1]
//'  x2 <- x[2]
//'  100 * (x2 - x1 * x1)^2 + (1 - x1)^2
//'}
//'grr <- function(x) { ## Gradient of 'fr'
//'  x1 <- x[1]
//'  x2 <- x[2]
//'  c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
//'    200 *      (x2 - x1 * x1))
//'}
//'
//'optim(c(-1.2,1), fr)
//'
//'## These do not converge in the default number of steps
//'optim(c(-1.2,1), fr, grr, method = "CG")
//'optim(c(-1.2,1), fr, grr, method = "CG", control = list(type = 2))
//'
//'optim(c(-1.2,1), fr, grr, method = "L-BFGS-B")
//'
//'optim(c(-1.2,1), fr, method = "SANN")
//'
//'## corresponding C++ implementation:
//'example1_rosen_other_methods()
//'@export
// [[Rcpp::export]]
void example1_rosen_other_methods() {
  Rosen rb;
  arma::vec x;

  // "Nelder-Mead": converged
  Roptim<Rosen> opt1;
  x = {-1.2, 1};
  opt1.minimize(rb, x);
  opt1.print();

  // "CG": did not converge in the default number of steps
  Roptim<Rosen> opt2("CG");
  x = {-1.2, 1};
  opt2.minimize(rb, x);
  opt2.print();

  // "CG": did not converge in the default number of steps
  Roptim<Rosen> opt3("CG");
  opt3.control.type = 2;
  x = {-1.2, 1};
  opt3.minimize(rb, x);
  opt3.print();

  // "L-BFGS-B"
  Roptim<Rosen> opt4("L-BFGS-B");
  x = {-1.2, 1};
  opt4.minimize(rb, x);
  opt4.print();

  // "SANN"
  Roptim<Rosen> opt5("SANN");
  x = {-1.2, 1};
  opt5.minimize(rb, x);
  opt5.print();
}

//'@title Example 1: Gradient/Hessian checks for the implemented C++ class of Rosenbrock function
//'@description Gradient/Hessian checks for the implemented C++ class of Rosenbrock function.
//'@export
// [[Rcpp::export]]
void example1_rosen_grad_hess_check() {
  Rosen rb;
  arma::vec x = {-1.2, 1};

  arma::vec grad1, grad2;
  rb.Gradient(x, grad1);
  rb.ApproximateGradient(x, grad2);

  arma::mat hess1, hess2;
  rb.Hessian(x, hess1);
  rb.ApproximateHessian(x, hess2);

  Rcpp::Rcout << "Gradient checking" << std::endl;
  grad1.t().print("analytic:");
  grad2.t().print("approximate:");

  Rcpp::Rcout << "-------------------------" << std::endl;

  Rcpp::Rcout << "Hessian checking" << std::endl;
  hess1.print("analytic:");
  hess2.print("approximate:");
}

//'@title Example 1: Minimize Rosenbrock function (with numerical gradient) using BFGS
//'@description Minimize Rosenbrock function (with numerical gradient) using BFGS.
//'@examples
//'fr <- function(x) {   ## Rosenbrock Banana function
//'  x1 <- x[1]
//'  x2 <- x[2]
//'  100 * (x2 - x1 * x1)^2 + (1 - x1)^2
//'}
//'
//'optim(c(-1.2,1), fr, NULL, method = "BFGS")
//'
//'## corresponding C++ implementation:
//'example1_rosen_nograd_bfgs()
//'@export
// [[Rcpp::export]]
void example1_rosen_nograd_bfgs() {
  RosenNoGrad rb;
  Roptim<RosenNoGrad> opt("BFGS");

  arma::vec x = {-1.2, 1};
  opt.minimize(rb, x);

  opt.par().t().print("par = ");
}

//////////////////////////////////////////////////
// EXAMPLE 2
//////////////////////////////////////////////////

class TSP : public Functor {
 public:
  TSP(const arma::mat &distmat) : distmat_(distmat) {
    os.sann_use_custom_function_ = true;
  }

  double operator()(const arma::vec &sq) override {
    arma::uvec idx1(sq.size() - 1);
    arma::uvec idx2(sq.size() - 1);

    std::copy(sq.cbegin(), sq.cend() - 1, idx1.begin());
    std::copy(sq.cbegin() + 1, sq.cend(), idx2.begin());

    idx1.for_each([](arma::uvec::elem_type &val) { val -= 1.0; });
    idx2.for_each([](arma::uvec::elem_type &val) { val -= 1.0; });

    arma::vec distvec(sq.size() - 1);
    for (int idx = 0; idx != distmat_.n_rows; ++idx) {
      distvec(idx) = distmat_(idx1(idx), idx2(idx));
    }

    return arma::sum(distvec);
  }

  void Gradient(const arma::vec &sq, arma::vec &grad) override {
    grad = sq;

    arma::vec idx =
        arma::linspace(2, distmat_.n_rows - 1, distmat_.n_rows - 2);
    arma::vec idx_shuffled = arma::shuffle(idx);
    idx_shuffled.for_each([](arma::vec::elem_type &val) { val -= 1.0; });

    arma::vec changepoints = idx_shuffled.subvec(0, 1);

    grad(changepoints(0)) = sq(changepoints(1));
    grad(changepoints(1)) = sq(changepoints(0));
  }

 private:
  arma::mat distmat_;
};

//'@title Example 1: Solve Travelling Salesman Problem (TSP) using SANN
//'@description Solve Travelling Salesman Problem (TSP) using SANN.
//'@param distmat a distance matrix for storing all pair of locations.
//'@param x initial route.
//'@examples
//'## Combinatorial optimization: Traveling salesman problem
//'library(stats) # normally loaded
//'
//'eurodistmat <- as.matrix(eurodist)
//'
//'distance <- function(sq) {  # Target function
//'  sq2 <- embed(sq, 2)
//'  sum(eurodistmat[cbind(sq2[,2], sq2[,1])])
//'}
//'
//'genseq <- function(sq) {  # Generate new candidate sequence
//'  idx <- seq(2, NROW(eurodistmat)-1)
//'  changepoints <- sample(idx, size = 2, replace = FALSE)
//'  tmp <- sq[changepoints[1]]
//'  sq[changepoints[1]] <- sq[changepoints[2]]
//'  sq[changepoints[2]] <- tmp
//'  sq
//'}
//'
//'sq <- c(1:nrow(eurodistmat), 1)  # Initial sequence: alphabetic
//'distance(sq)
//'# rotate for conventional orientation
//'loc <- -cmdscale(eurodist, add = TRUE)$points
//'x <- loc[,1]; y <- loc[,2]
//'s <- seq_len(nrow(eurodistmat))
//'tspinit <- loc[sq,]
//'
//'plot(x, y, type = "n", asp = 1, xlab = "", ylab = "",
//'     main = "initial solution of traveling salesman problem", axes = FALSE)
//'arrows(tspinit[s,1], tspinit[s,2], tspinit[s+1,1], tspinit[s+1,2],
//'       angle = 10, col = "green")
//'text(x, y, labels(eurodist), cex = 0.8)
//'
//'## The original R optimization:
//'## set.seed(123) # chosen to get a good soln relatively quickly
//'## res <- optim(sq, distance, genseq, method = "SANN",
//'##              control = list(maxit = 30000, temp = 2000, trace = TRUE,
//'##              REPORT = 500))
//'## res  # Near optimum distance around 12842
//'
//'## corresponding C++ implementation:
//'set.seed(10)  # chosen to get a good soln relatively quickly
//'res <- example2_tsp_sann(eurodistmat, sq)
//'
//'tspres <- loc[res$par,]
//'plot(x, y, type = "n", asp = 1, xlab = "", ylab = "",
//'     main = "optim() 'solving' traveling salesman problem", axes = FALSE)
//'arrows(tspres[s,1], tspres[s,2], tspres[s+1,1], tspres[s+1,2],
//'       angle = 10, col = "red")
//'text(x, y, labels(eurodist), cex = 0.8)
//'@export
// [[Rcpp::export]]
Rcpp::List example2_tsp_sann(arma::mat distmat, arma::vec x) {

  TSP dist(distmat);
  Roptim<TSP> opt("SANN");
  opt.control.maxit = 30000;
  opt.control.temp = 2000;
  opt.control.trace = true;
  opt.control.REPORT = 500;

  opt.minimize(dist, x);

  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();

  return Rcpp::List::create(Rcpp::Named("par") = x);
}

//////////////////////////////////////////////////
// EXAMPLE 3
//////////////////////////////////////////////////

class Flb : public Functor {
 public:
  double operator()(const arma::vec &x) override {
    int p = x.size();

    arma::vec part1 = arma::ones<arma::vec>(p) * 4;
    part1(0) = 1;

    arma::vec tmp = arma::ones<arma::vec>(p);
    std::copy(x.cbegin(), x.cend() - 1, tmp.begin() + 1);

    arma::vec part2 = arma::pow(x - arma::pow(tmp, 2), 2);

    return arma::dot(part1, part2);
  }
};

//'@title Example 3: Minimize a function using L-BFGS-B with 25-dimensional box constrained
//'@description Minimize a function using L-BFGS-B with 25-dimensional box constrained.
//'@examples
//'flb <- function(x)
//'{ p <- length(x); sum(c(1, rep(4, p-1)) * (x - c(1, x[-p])^2)^2) }
//'## 25-dimensional box constrained
//'optim(rep(3, 25), flb, NULL, method = "L-BFGS-B",
//'      lower = rep(2, 25), upper = rep(4, 25)) # par[24] is *not* at boundary
//'
//' ## corresponding C++ implementation:
//' example3_flb_25_dims_box_con()
//'@export
// [[Rcpp::export]]
void example3_flb_25_dims_box_con() {
  Flb f;
  arma::vec lower = arma::ones<arma::vec>(25) * 2;
  arma::vec upper = arma::ones<arma::vec>(25) * 4;

  Roptim<Flb> opt("L-BFGS-B");
  opt.set_lower(lower);
  opt.set_upper(upper);
  opt.control.trace = 1;

  arma::vec x = arma::ones<arma::vec>(25) * 3;
  opt.minimize(f, x);

  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();
}

//////////////////////////////////////////////////
// EXAMPLE 4
//////////////////////////////////////////////////

class Fw : public Functor {
 public:
  double operator()(const arma::vec &xval) override {
    double x = arma::as_scalar(xval);

    return 10 * std::sin(0.3 * x) * std::sin(1.3 * std::pow(x, 2.0)) +
           0.00001 * std::pow(x, 4.0) + 0.2 * x + 80;
  }
};

//'@title Example 4: Minimize a "wild" function using SANN and BFGS
//'@description Minimize a "wild" function using SANN and BFGS.
//'@examples
//'## "wild" function , global minimum at about -15.81515
//'fw <- function (x)
//'  10*sin(0.3*x)*sin(1.3*x^2) + 0.00001*x^4 + 0.2*x+80
//'plot(fw, -50, 50, n = 1000, main = "optim() minimising 'wild function'")
//'
//'res <- optim(50, fw, method = "SANN",
//'             control = list(maxit = 20000, temp = 20, parscale = 20))
//'res
//'## Now improve locally {typically only by a small bit}:
//'(r2 <- optim(res$par, fw, method = "BFGS"))
//'points(r2$par,  r2$value,  pch = 8, col = "red", cex = 2)
//'
//' ## corresponding C++ implementation:
//' example4_wild_fun()
//'@export
// [[Rcpp::export]]
void example4_wild_fun() {
  // "wild" function , global minimum at about -15.81515
  Fw f;

  Roptim<Fw> opt("SANN");
  opt.control.maxit = 20000;
  opt.control.temp = 20;
  opt.control.parscale = 20;
  opt.control.trace = 1;

  arma::vec x = {50};
  opt.minimize(f, x);
  x.print();

  // Now improve locally {typically only by a small bit}:
  Roptim<Fw> opt2("BFGS");
  opt2.minimize(f, x);
  x.print();
}
