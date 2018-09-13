#include <RcppArmadillo.h>
#include "roptim.h"

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

// [[Rcpp::export]]
void example1_rosen_bfgs()
{
  Rosen rb;
  Roptim<Rosen> opt("BFGS");
  opt.control.trace = 1;
  opt.set_hessian(true);

  arma::vec x = {-1.2, 1};
  opt.minimize(rb, x);

  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();
}

// [[Rcpp::export]]
void example1_rosen_other_methods()
{
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
}

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

// [[Rcpp::export]]
void example1_rosen_nograd_bfgs()
{
  RosenNoGrad rb;
  Roptim<RosenNoGrad> opt("BFGS");

  arma::vec x = {-1.2, 1};
  opt.minimize(rb, x);

  opt.par().t().print("par = ");
}

// [[Rcpp::export]]
Rcpp::List rcpp_hello() {
  Rcpp::CharacterVector x = Rcpp::CharacterVector::create("foo", "bar");
  Rcpp::NumericVector y   = Rcpp::NumericVector::create(0.0, 1.0);
  Rcpp::List z            = Rcpp::List::create(x, y);
  return z;
}
