// Copyright (c) 2018 Yi Pan <ypan1988@gmail.com>

#ifndef ROPTIM_H_
#define ROPTIM_H_

#include <cassert>
#include <cstddef>

#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <RcppArmadillo.h>

#include "applic.h"
#include "functor.h"
#include "samin.h"

namespace roptim {

template <typename Derived>
class Roptim {
 public:
  std::string method_;
  arma::vec lower_, upper_;
  bool hessian_flag_ = false;
  arma::mat hessian_;

  arma::vec lower() const { return lower_; }
  arma::vec upper() const { return upper_; }

  arma::vec par() const { return par_; }
  double value() const { return val_; }
  int fncount() const { return fncount_; }
  int grcount() const { return grcount_; }
  int convergence() const { return fail_; }
  std::string message() const { return message_; }
  arma::mat hessian() const { return hessian_; }

  void print() const {
    par_.t().print(".par()");
    Rcpp::Rcout << "\n.value()\n" << val_ << std::endl;
    Rcpp::Rcout << "\n.fncount()\n" << fncount_ << std::endl;
    Rcpp::Rcout << "\n.grcount()\n" << grcount_ << std::endl;
    Rcpp::Rcout << "\n.convergence()\n" << fail_ << std::endl;
    Rcpp::Rcout << "\n.message()\n" << message_ << std::endl;
    if (hessian_flag_)
      hessian_.print("\n.hessian()");
    Rcpp::Rcout << std::endl;
  }

 private:
  arma::vec par_;
  double val_ = 0.0;
  int fncount_ = 0;
  int grcount_ = 0;
  int fail_ = 0;
  std::string message_ = "NULL";

 public:
  struct RoptimControl {
    std::size_t trace = 0;
    double fnscale = 1.0;
    arma::vec parscale;
    arma::vec ndeps;
    std::size_t maxit = 100;
    double abstol = R_NegInf;
    double reltol = sqrt(2.220446e-16);
    double alpha = 1.0;
    double beta = 0.5;
    double gamma = 2.0;
    int REPORT = 10;
    bool warn_1d_NelderMead = true;
    int type = 1;
    int lmm = 5;
    double factr = 1e7;
    double pgtol = 0.0;
    double temp = 10.0;
    int tmax = 10;
  } control;

  Roptim(const std::string method = "Nelder-Mead") : method_(method) {
    if (method_ != "Nelder-Mead" && method_ != "BFGS" && method_ != "CG" &&
        method_ != "L-BFGS-B" && method_ != "SANN")
      Rcpp::stop("Roptim(): unsupported method");

    if (method_ == "Nelder-Mead") control.maxit = 500;
    if (method_ == "SANN") {
      control.maxit = 10000;
      control.REPORT = 100;
    }
  }

  void set_method(const std::string &method) {
    if (method != "Nelder-Mead" && method != "BFGS" && method != "CG" &&
        method != "L-BFGS-B" && method != "SANN")
      Rcpp::stop("set_method(): unsupported method");
    else
      method_ = method;

    if (method_ == "Nelder-Mead") {
      control.maxit = 500;
      control.REPORT = 10;
    } else if (method_ == "SANN") {
      control.maxit = 10000;
      control.REPORT = 100;
    } else {
      control.maxit = 100;
      control.REPORT = 10;
    }
  }

  void set_lower(const arma::vec &lower) {
    if (method_ != "L-BFGS-B")
      Rcpp::warning(
          "Roptim::set_lower(): bounds can only be used with method L-BFGS-B");
    method_ = "L-BFGS-B";
    lower_ = lower;
  }

  void set_upper(const arma::vec &upper) {
    if (method_ != "L-BFGS-B")
      Rcpp::warning(
          "Roptim::set_upper(): bounds can only be used with method L-BFGS-B");
    method_ = "L-BFGS-B";
    upper_ = upper;
  }

  void set_hessian(bool flag) {
    hessian_flag_ = flag;
  }

  void minimize(Derived &func, arma::vec &par) {
    int debug = 0;

    // TODO: check if lower and upper is used

    int npar = par.size();
    if (lower_.is_empty()) {
      lower_ = arma::zeros<arma::vec>(npar);
      lower_.for_each([](arma::mat::elem_type &val) { val = R_NegInf; });
    }
    if (upper_.is_empty()) {
      upper_ = arma::zeros<arma::vec>(npar);
      upper_.for_each([](arma::mat::elem_type &val) { val = R_PosInf; });
    }

    if (control.parscale.is_empty())
      control.parscale = arma::ones<arma::vec>(npar);
    if (control.ndeps.is_empty())
      control.ndeps = arma::ones<arma::vec>(npar) * 1e-3;

    arma::vec dpar = arma::zeros<arma::vec>(npar);
    arma::vec opar = arma::zeros<arma::vec>(npar);
    dpar = par / control.parscale;

    func.os.ndeps_ = control.ndeps;
    func.os.fnscale_ = control.fnscale;
    func.os.parscale_ = control.parscale;

    if (method_ == "Nelder-Mead") {
      if (debug) std::cout << "Nelder-Mead:" << std::endl;
      nmmin(npar, dpar.memptr(), opar.memptr(), &val_, fminfn, &fail_,
            control.abstol, control.reltol, &func, control.alpha, control.beta,
            control.gamma, control.trace, &fncount_, control.maxit);

      par = opar % control.parscale;
    } else if (method_ == "SANN") {
      if (debug) std::cout << "SANN:" << std::endl;

      int trace = 0;
      if (control.trace) trace = control.REPORT;

      internal::samin(npar, dpar.memptr(), &val_, fminfn,
                               control.maxit, control.tmax, control.temp, trace,
                               &func);
      par = dpar % control.parscale;
      fncount_ = npar > 0 ? control.maxit : 1;
    } else if (method_ == "BFGS") {
      if (debug) std::cout << "BFGS:" << std::endl;

      arma::ivec mask = arma::ones<arma::ivec>(npar);
      vmmin(npar, dpar.memptr(), &val_, fminfn, fmingr,
            control.maxit, control.trace, mask.memptr(), control.abstol,
            control.reltol, control.REPORT, &func, &fncount_, &grcount_,
            &fail_);

      par = dpar % control.parscale;
    } else if (method_ == "CG") {
      cgmin(npar, dpar.memptr(), opar.memptr(), &val_, fminfn,
            fmingr, &fail_, control.abstol, control.reltol, &func,
            control.type, control.trace, &fncount_, &grcount_, control.maxit);

      par = opar % control.parscale;
    } else if (method_ == "L-BFGS-B") {
      arma::ivec nbd = arma::zeros<arma::ivec>(npar);
      char msg[60];

      func.os.lower_ = lower_;
      func.os.upper_ = upper_;

      lbfgsb(npar, control.lmm, dpar.memptr(), lower_.memptr(), upper_.memptr(),
             nbd.memptr(), &val_, fminfn, fmingr, &fail_,
             &func, control.factr, control.pgtol, &fncount_, &grcount_,
             control.maxit, msg, control.trace, control.REPORT);

      message_ = msg;
      par = dpar % control.parscale;
    }

    par_ = par;

    if (hessian_flag_) func.ApproximateHessian(par_, hessian_);
  }
};

}  // namespace roptim

#endif  // ROPTIM_H_
