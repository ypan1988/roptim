// Copyright (c) 2018 Yi Pan <ypan1988@gmail.com>

#ifndef FUNCTOR_H_
#define FUNCTOR_H_

#include <RcppArmadillo.h>

namespace roptim {

struct OptStruct {
  bool has_grad_;
  bool has_hess_;
  arma::vec ndeps_;       // tolerances for numerical derivatives
  double fnscale_ = 1.0;  // scaling for objective
  arma::vec parscale_;    // scaling for parameters
  int usebounds_ = 0;
  arma::vec lower_, upper_;
};

//template <typename Derived>
class Functor {
 public:
  Functor() { //UpdateOptStruct();
  }

  virtual ~Functor() {}

  virtual double operator()(const arma::vec &par) = 0;
  virtual void Gradient(const arma::vec &par, arma::vec &grad) {
    ApproximateGradient(par, grad);
  }
  virtual void Hessian(const arma::vec &par, arma::mat &hess) {
    ApproximateHessian(par, hess);
  }

  // Returns forward-difference approximation of Gradient
  void ApproximateGradient(const arma::vec &par, arma::vec &grad);

  // Returns forward-difference approximation of Hessian
  void ApproximateHessian(const arma::vec &par, arma::mat &hess);

  OptStruct os;

 // private:
 //  void UpdateOptStruct() {
 //    if (&Derived::Gradient == &Functor::Gradient)
 //      os.has_grad_ = false;
 //    else
 //      os.has_grad_ = true;
 //
 //    if (&Derived::Hessian == &Functor::Hessian)
 //      os.has_hess_ = false;
 //    else
 //      os.has_hess_ = true;
 //  }
};

// template <typename Derived>
// inline void Functor<Derived>::ApproximateGradient(const arma::vec &par,
//                                                   arma::vec &grad) {
inline void Functor::ApproximateGradient(const arma::vec &par, arma::vec &grad) {
  if (os.parscale_.is_empty())
    os.parscale_ = arma::ones<arma::vec>(par.size());
  if (os.ndeps_.is_empty())
    os.ndeps_ = arma::ones<arma::vec>(par.size()) * 1e-3;

  grad = arma::zeros<arma::vec>(par.size());
  arma::vec x = par % os.parscale_;

  if (os.usebounds_ == 0) {
    for (std::size_t i = 0; i != par.size(); ++i) {
      double eps = os.ndeps_(i);

      x(i) = (par(i) + eps) * os.parscale_(i);
      double val1 = operator()(x) / os.fnscale_;

      x(i) = (par(i) - eps) * os.parscale_(i);
      double val2 = operator()(x) / os.fnscale_;

      grad(i) = (val1 - val2) / (2 * eps);

      x(i) = par(i) * os.parscale_(i);
    }
  } else { // use bounds
    for (std::size_t i = 0; i != par.size(); ++i) {
      double epsused = os.ndeps_(i);
      double eps = os.ndeps_(i);

      double tmp = par(i) + eps;
      if (tmp > os.upper_(i)) {
        tmp = os.upper_(i);
        epsused = tmp - par(i);
      }

      x(i) = tmp * os.parscale_(i);
      double val1 = operator()(x) / os.fnscale_;

      tmp = par(i) - eps;
      if (tmp < os.lower_(i)) {
        tmp = os.lower_(i);
        eps = par(i) - tmp;
      }

      x(i) = tmp * os.parscale_(i);
      double val2 = operator()(x) / os.fnscale_;

      grad(i) = (val1 - val2) / (epsused + eps);

      x(i) = par(i) * os.parscale_(i);
    }
  }
}

// template <typename Derived>
// inline void Functor<Derived>::ApproximateHessian(const arma::vec &par,
//                                                  arma::mat &hess) {
inline void Functor::ApproximateHessian(const arma::vec &par, arma::mat &hess) {
  if (os.parscale_.is_empty())
    os.parscale_ = arma::ones<arma::vec>(par.size());
  if (os.ndeps_.is_empty())
    os.ndeps_ = arma::ones<arma::vec>(par.size()) * 1e-3;

  hess = arma::zeros<arma::mat>(par.size(), par.size());
  arma::vec dpar = par / os.parscale_;
  arma::vec df1 = arma::zeros<arma::vec>(par.size());
  arma::vec df2 = arma::zeros<arma::vec>(par.size());

  for (std::size_t i = 0; i != par.size(); ++i) {
    double eps = os.ndeps_(i) / os.parscale_(i);
    dpar(i) += eps;
    Gradient(dpar, df1);
    dpar(i) -= 2 * eps;
    Gradient(dpar, df2);
    for (std::size_t j = 0; j != par.size(); ++j)
      hess(i, j) = os.fnscale_ * (df1(j) - df2(j)) /
        (2 * eps * os.parscale_(i) * os.parscale_(j));
    dpar(i) = dpar(i) + eps;
  }

  // now symmetrize
  for (std::size_t i = 0; i != par.size(); ++i) {
    for (std::size_t j = 0; j != par.size(); ++j) {
      double tmp = 0.5 * (hess(i, j) + hess(j, i));

      hess(i, j) = tmp;
      hess(j, i) = tmp;
    }
  }
  // hess = arma::zeros<arma::mat>(par.size(), par.size());

  // const double kEpsilon = 1.0e-8;
  // arma::vec par_h = par;
  // arma::vec grad_old;
  // Gradient(par, grad_old);
  // for (arma::uword j = 0; j != par.size(); ++j) {
  //   double temp = par_h(j);
  //   double h = kEpsilon * abs(temp);
  //   if (h == 0) h = kEpsilon;
  //   par_h(j) = temp + h;
  //   h = par_h(j) - temp;
  //   arma::vec grad_h;
  //   Gradient(par_h, grad_h);
  //   par_h = temp;
  //   for (arma::uword i = 0; i != par.size(); ++i) {
  //     hess(i, j) = (grad_h(i) - grad_old(i)) / h;
  //   }
  // }
}

// template <typename Derived>
inline double fminfn(int n, double *x, void *ex) {
  //OptStruct os(static_cast<Derived *>(ex)->os);
  OptStruct os(static_cast<Functor *>(ex)->os);

  arma::vec par(x, n);
  par %= os.parscale_;
  //return static_cast<Derived *>(ex)->operator()(par) / os.fnscale_;
  return static_cast<Functor *>(ex)->operator()(par) / os.fnscale_;
}

//template <typename Derived>
inline void fmingr(int n, double *x, double *gr, void *ex) {
  // OptStruct os(static_cast<Derived *>(ex)->os);
  OptStruct os(static_cast<Functor *>(ex)->os);

  arma::vec par(x, n), grad;
  par %= os.parscale_;
  //static_cast<Derived *>(ex)->Gradient(par, grad);
  static_cast<Functor *>(ex)->Gradient(par, grad);
  for (auto i = 0; i != n; ++i)
    gr[i] = grad(i) * (os.parscale_(i) / os.fnscale_);
}

}  // namespace roptim

#endif
