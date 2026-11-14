library(testthat)

# testthat runs with the working directory set to r/tests; fall back to the project root
root <- if (file.exists("../../data/processed/segments_long.csv")) normalizePath("../..") else normalizePath(".")
source(file.path(root, "r", "chainladder.R"))

test_that("deterministic triangle is reproduced exactly", {
  f <- c(2, 1.5, 1.2, 1.1)
  first <- c(100, 120, 90, 110, 130)
  sq <- outer(first, c(1, cumprod(f)))
  tri <- sq
  tri[row(tri) + col(tri) > 6] <- NA
  cl <- chain_ladder(tri)
  expect_equal(cl$ldf, f)
  expect_equal(unname(cl$ultimate), sq[, 5])
})

test_that("R and Python give the same LDFs and Mack s.e. on the workers' comp paid triangle", {
  tri <- read_segment_triangle(file.path(root, "data/processed/segments_long.csv"), "wkcomp")
  py_ldf <- read.csv(file.path(root, "reports/tables/development_factors.csv"))
  py_ldf <- py_ldf[py_ldf$segment == "wkcomp" & py_ldf$basis == "paid", "ldf"]
  expect_equal(volume_weighted_ldf(tri), py_ldf, tolerance = 1e-5)

  tails <- read.csv(file.path(root, "reports/tables/tail_selection.csv"))
  tail <- tails$selected_paid_tail[tails$segment == "wkcomp"]
  by_ay <- read.csv(file.path(root, "reports/tables/by_accident_year_wkcomp.csv"))
  cl <- chain_ladder(tri, tail = tail)
  expect_equal(unname(cl$ultimate), by_ay$ult_paid_cl, tolerance = 1e-5)
  # Python reports Mack s.e. scaled by the constant tail factor
  expect_equal(unname(mack_se(tri)$se) * tail, by_ay$mack_se_paid, tolerance = 1e-5)
})
