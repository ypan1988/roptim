#' roptim
#'
#' This package provides a unified wrapper interface to the C codes of
#' the five optimization algorithms ("Nelder-Mead", "BFGS", "CG", "L-BFGS-B"
#' and "SANN") underlying optim function and enables users performing general
#' purpose optimization tasks using C++ without reimplementing the optimization
#' routines.
#'
#' @docType package
#' @author Yi Pan
#' @import Rcpp
#' @importFrom Rcpp evalCpp
#' @useDynLib roptim
#' @name roptim
NULL
