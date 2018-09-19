#include <algorithm>
#include <cmath>

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

// [[Rcpp::export]]
Rcpp::List example2_sann_tsp(arma::mat eurodistmat, arma::vec x) {

  TSP dist(eurodistmat);
  Roptim<TSP> opt("SANN");
  opt.control.maxit = 30000;
  opt.control.temp = 2000;
  opt.control.trace = true;
  opt.control.REPORT = 500;

  // arma::vec x =
  //     arma::linspace(1, eurodistmat.n_rows + 1, eurodistmat.n_rows + 1);
  // x(eurodistmat.n_rows) = 1;

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

// [[Rcpp::export]]
void example1_rosen_sann() {
  Rosen rb;
  arma::vec x;

  Roptim<Rosen> opt1("SANN");
  opt1.control.trace = 1;
  x = {-1.2, 1};
  opt1.minimize(rb, x);
  opt1.print();
}

// [[Rcpp::export]]
Rcpp::List rcpp_hello() {
  Rcpp::CharacterVector x = Rcpp::CharacterVector::create("foo", "bar");
  Rcpp::NumericVector y = Rcpp::NumericVector::create(0.0, 1.0);
  Rcpp::List z = Rcpp::List::create(x, y);
  return z;
}
