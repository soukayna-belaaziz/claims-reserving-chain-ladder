# Base-R re-implementation of the volume-weighted chain ladder and Mack (1993)
# standard errors, used to cross-check the Python results on one triangle.
#
# Usage (from the project root):  Rscript r/chainladder.R
# Requires only base R (testthat for the tests in r/tests).

read_segment_triangle <- function(path, segment, value = "CumPaidLoss", valuation_year = 2007) {
  long <- read.csv(path, stringsAsFactors = FALSE)
  long <- long[long$segment == segment, ]
  origins <- sort(unique(long$AccidentYear))
  lags <- sort(unique(long$DevelopmentLag))
  tri <- matrix(NA_real_, length(origins), length(lags), dimnames = list(origins, lags * 12))
  for (r in seq_len(nrow(long))) {
    if (long$DevelopmentYear[r] <= valuation_year) {
      tri[as.character(long$AccidentYear[r]), long$DevelopmentLag[r]] <- long[[value]][r]
    }
  }
  tri
}

volume_weighted_ldf <- function(tri) {
  n <- ncol(tri)
  sapply(seq_len(n - 1), function(j) {
    ok <- !is.na(tri[, j]) & !is.na(tri[, j + 1])
    sum(tri[ok, j + 1]) / sum(tri[ok, j])
  })
}

latest_lag <- function(tri) apply(tri, 1, function(r) max(which(!is.na(r))))

project_square <- function(tri, ldf) {
  out <- tri
  for (j in seq_len(ncol(tri) - 1)) {
    miss <- is.na(out[, j + 1]) & !is.na(out[, j])
    out[miss, j + 1] <- out[miss, j] * ldf[j]
  }
  out
}

chain_ladder <- function(tri, tail = 1) {
  ldf <- volume_weighted_ldf(tri)
  full <- project_square(tri, ldf)
  latest <- tri[cbind(seq_len(nrow(tri)), latest_lag(tri))]
  ultimate <- full[, ncol(full)] * tail
  list(ldf = ldf, tail = tail, latest = latest, ultimate = ultimate,
       reserve = ultimate - latest, full = full)
}

mack_sigma2 <- function(tri, ldf) {
  n <- ncol(tri)
  s2 <- rep(NA_real_, n - 1)
  for (k in seq_len(n - 1)) {
    ok <- !is.na(tri[, k]) & !is.na(tri[, k + 1])
    m <- sum(ok)
    if (m > 1) {
      f_ik <- tri[ok, k + 1] / tri[ok, k]
      s2[k] <- sum(tri[ok, k] * (f_ik - ldf[k])^2) / (m - 1)
    }
  }
  last <- n - 1
  if (is.na(s2[last])) s2[last] <- min(s2[last - 1]^2 / s2[last - 2], s2[last - 2], s2[last - 1])
  s2
}

mack_se <- function(tri) {
  n <- nrow(tri)
  cl <- chain_ladder(tri)
  ldf <- cl$ldf
  s2 <- mack_sigma2(tri, ldf)
  s_k <- sapply(seq_len(n - 1), function(k) sum(tri[seq_len(n - k), k]))
  ult <- cl$full[, n]
  lag <- latest_lag(tri)
  mse <- numeric(n)
  cov_term <- 0
  for (i in seq_len(n)) {
    if (lag[i] < n) {
      ks <- lag[i]:(n - 1)
      mse[i] <- ult[i]^2 * sum(s2[ks] / ldf[ks]^2 * (1 / cl$full[i, ks] + 1 / s_k[ks]))
      if (i < n) cov_term <- cov_term + ult[i] * sum(ult[(i + 1):n]) * sum(2 * s2[ks] / ldf[ks]^2 / s_k[ks])
    }
  }
  list(se = sqrt(mse), total_se = sqrt(sum(mse) + cov_term))
}

if (sys.nframe() == 0) {
  tri <- read_segment_triangle("data/processed/segments_long.csv", "wkcomp")
  cl <- chain_ladder(tri, tail = 1)
  mk <- mack_se(tri)
  cat("Workers' comp (State Farm), paid, valuation 2007, no tail\n")
  print(round(cl$ldf, 4))
  res <- data.frame(accident_year = rownames(tri), latest = cl$latest,
                    ultimate = round(cl$ultimate), reserve = round(cl$reserve), mack_se = round(mk$se))
  print(res, row.names = FALSE)
  cat(sprintf("Total reserve %.0f, Mack s.e. %.0f\n", sum(cl$reserve), mk$total_se))
}
