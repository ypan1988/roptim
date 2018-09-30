#' roptim
#'
#' Perform general purpose optimization in R using C++. A unified wrapper
#' interface is provided to call C functions of the five optimization algorithms
#' ('Nelder-Mead', 'BFGS', 'CG', 'L-BFGS-B' and 'SANN') underlying optim().
#'
#' @docType package
#' @author Yi Pan
#' @import Rcpp
#' @importFrom Rcpp evalCpp
#' @useDynLib roptim
#' @name roptim
NULL
