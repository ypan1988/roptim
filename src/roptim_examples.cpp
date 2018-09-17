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

class Distance : public Functor {
 public:
  Distance(const arma::mat &eurodistmat) : eurodistmat_(eurodistmat) {
    os.has_grad_ = 1;
  }

  double operator()(const arma::vec &sq) override {
    arma::uvec idx1(sq.size() - 1);
    arma::uvec idx2(sq.size() - 1);

    std::copy(sq.cbegin(), sq.cend() - 1, idx1.begin());
    std::copy(sq.cbegin() + 1, sq.cend(), idx2.begin());

    idx1.for_each([](arma::uvec::elem_type &val) { val -= 1.0; });
    idx2.for_each([](arma::uvec::elem_type &val) { val -= 1.0; });

    arma::vec dist_vec(sq.size() - 1);
    for (int idx = 0; idx != eurodistmat_.n_rows; ++idx) {
      dist_vec(idx) = eurodistmat_(idx1(idx), idx2(idx));
    }

    return arma::sum(dist_vec);
  }

  void Gradient(const arma::vec &sq, arma::vec &grad) override {
    grad = sq;

    arma::vec idx =
        arma::linspace(2, eurodistmat_.n_rows - 1, eurodistmat_.n_rows - 2);
    arma::vec idx_shuffled = arma::shuffle(idx);
    idx_shuffled.for_each([](arma::vec::elem_type &val) { val -= 1.0; });

    arma::vec changepoints = idx_shuffled.subvec(0, 1);

    grad(changepoints(0)) = sq(changepoints(1));
    grad(changepoints(1)) = sq(changepoints(0));
  }

 private:
  arma::mat eurodistmat_;
};

// [[Rcpp::export]]
void example2_travelling_salesman() {
  arma::mat eurodistmat = {
      {0,    3313, 2963, 3175, 3339, 2762, 3276, 2610, 4485, 2977, 3030,
       4532, 2753, 3949, 2865, 2282, 2179, 3000, 817,  3927, 1991},
      {3313, 0,   1318, 1326, 1294, 1498, 2218, 803,  1172, 2018, 1490,
       1305, 645, 636,  521,  1014, 1365, 1033, 1460, 2868, 1802},
      {2963, 1318, 0,    204,  583, 206, 966, 677,  2256, 597, 172,
       2084, 690,  1558, 1011, 925, 747, 285, 1511, 1616, 1175},
      {3175, 1326, 204,  0,    460,  409, 1136, 747,  2224, 714, 330,
       2052, 739,  1550, 1059, 1077, 977, 280,  1662, 1786, 1381},
      {3339, 1294, 583,  460,  0,    785,  1545, 853,  2047, 1115, 731,
       1827, 789,  1347, 1101, 1209, 1160, 340,  1794, 2196, 1588},
      {2762, 1498, 206,  409,  785, 0,   760, 1662, 2436, 460, 269,
       2290, 714,  1764, 1035, 911, 583, 465, 1497, 1403, 937},
      {3276, 2218, 966,  1136, 1545, 760,  0,    1418, 3196, 460, 269,
       2971, 1458, 2498, 1778, 1537, 1104, 1176, 2050, 650,  1455},
      {2610, 803, 677,  747, 853, 1662, 1418, 0,   1975, 1118, 895,
       1936, 158, 1439, 425, 328, 591,  513,  995, 2068, 1019},
      {4485, 1172, 2256, 2224, 2047, 2436, 3196, 1975, 0,    2897, 2428,
       676,  1817, 698,  1693, 2185, 2565, 1971, 2631, 3886, 2974},
      {2977, 2018, 597,  714,  1115, 460, 460, 1118, 2897, 0,   550,
       2671, 1159, 2198, 1479, 1238, 805, 877, 1751, 949,  1155},
      {3030, 1490, 172,  330,  731,  269, 269, 895,  2428, 550, 0,
       2280, 863,  1730, 1183, 1098, 851, 457, 1683, 1500, 1205},
      {4532, 1305, 2084, 2052, 1827, 2290, 2971, 1936, 676,  2671, 2280,
       0,    1178, 668,  1762, 2250, 2507, 1799, 2700, 3231, 2937},
      {2753, 645, 690,  739, 789, 714, 1458, 158,  1817, 1159, 863,
       1178, 0,   1281, 320, 328, 724, 471,  1048, 2108, 1157},
      {3949, 636,  1558, 1550, 1347, 1764, 2498, 1439, 698,  2198, 1730,
       668,  1281, 0,    1157, 1724, 2010, 1273, 2097, 3188, 2409},
      {2865, 521, 1011, 1059, 1101, 1035, 1778, 425,  1693, 1479, 1183,
       1762, 320, 1157, 0,    618,  1109, 792,  1011, 2428, 1363},
      {2282, 1014, 925,  1077, 1209, 911, 1537, 328, 2185, 1238, 1098,
       2250, 328,  1724, 618,  0,    331, 856,  586, 2187, 898},
      {2179, 1365, 747,  977,  1160, 583, 1104, 591, 2565, 805, 851,
       2507, 724,  2010, 1109, 331,  0,   821,  946, 1754, 428},
      {3000, 1033, 285,  280, 340, 465, 1176, 513,  1971, 877, 457,
       1799, 471,  1273, 792, 856, 821, 0,    1476, 1827, 1249},
      {817,  1460, 1511, 1662, 1794, 1497, 2050, 995, 2631, 1751, 1683,
       2700, 1048, 2097, 1011, 586,  946,  1476, 0,   2707, 1209},
      {3927, 2868, 1616, 1786, 2196, 1403, 650,  2068, 3886, 949, 1500,
       3231, 2108, 3188, 2428, 2187, 1754, 1827, 2707, 0,    2105},
      {1991, 1802, 1175, 1381, 1588, 937, 1455, 1019, 2974, 1155, 1205,
       2937, 1157, 2409, 1363, 898,  428, 1249, 1209, 2105, 0},
  };

  Distance dist(eurodistmat);
  Roptim<Distance> opt("SANN");
  opt.control.maxit = 30000;
  opt.control.temp = 2000;
  opt.control.trace = true;
  opt.control.REPORT = 500;

  arma::vec x =
      arma::linspace(1, eurodistmat.n_rows + 1, eurodistmat.n_rows + 1);
  x(eurodistmat.n_rows) = 1;

  opt.minimize(dist, x);

  Rcpp::Rcout << "-------------------------" << std::endl;
  opt.print();
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
