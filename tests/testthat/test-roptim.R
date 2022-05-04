test_that("BFGS", {

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
  res1 <- optim(c(-1.2,1), fr, grr, method = "BFGS", control = list(trace=FALSE), hessian = TRUE)
  res2 <- example1_rosen_bfgs(FALSE)

  expect_equal(c(res2$par), res1$par)
  expect_equal(res2$value, res1$value)
  expect_equal(res2$fncount, as.numeric(res1$counts[1]))
  expect_equal(res2$grcount, as.numeric(res1$counts[2]))
  expect_equal(res2$convergence, res1$convergence)
  expect_equal(res2$message, "NULL")
  expect_equal(res2$hessian, res1$hessian)
})
