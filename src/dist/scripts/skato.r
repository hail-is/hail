#! /usr/bin/env Rscript

# Setup sink so only JSON is returned in stdout
sink('stdout', type = c("output", "message"))

# load libraries
suppressPackageStartupMessages(library("jsonlite"))
suppressWarnings(library("SKAT"))

# read json data from stdin
args <- commandArgs(trailingOnly = TRUE)
json_data <-
  fromJSON(file("stdin"), simplifyVector = TRUE, simplifyMatrix = TRUE)

# get variables
Y <- json_data$Y
yType <- json_data$yType
nResampling <- json_data$nResampling
typeResampling <- json_data$typeResampling
adjustment <- json_data$adjustment
kernel <- json_data$kernel
method <- json_data$method
weightsBeta <- json_data$weightsBeta
imputeMethod <- json_data$imputeMethod
rCorr <- json_data$rCorr
missingCutoff <- json_data$missingCutoff
estimateMAF <- json_data$estimateMAF
seed <- json_data$seed

# Set random seed
set.seed(seed)

# calculate null model
if ("COV" %in% names(json_data)) {
  obj <- suppressWarnings(
    SKAT_Null_Model(
      Y ~ json_data$COV,
      out_type = yType,
      n.Resampling = nResampling,
      type.Resampling = typeResampling,
      Adjustment = adjustment
    )
  )
} else {
  obj <- suppressWarnings(
    SKAT_Null_Model(
      Y ~ 1,
      out_type = yType,
      n.Resampling = nResampling,
      type.Resampling = typeResampling,
      Adjustment = adjustment
    )
  )
}

group_data <- json_data$groups

runSkatO <- function(name, data) {
  Z <- t(data)
  
  res <- try(suppressWarnings(SKAT(
    Z, obj,
    kernel = kernel,
    method = method,
    weights.beta = weightsBeta,
    impute.method = imputeMethod,
    r.corr = rCorr,
    missing_cutoff = missingCutoff,
    estimate_MAF = estimateMAF
  )))
  
  if (class(res) == "try-error") {
    res <-
      list(
        p.value = NA, n.marker = NA, p.value.noadj = NA, n.marker.test = NA
      )
  }
  
  return(
    list(
      groupName = unbox(name),
      pValue = unbox(res$p.value),
      pValueNoAdj = unbox(res$p.value.noadj),
      nMarker = unbox(res$param$n.marker),
      nMarkerTest = unbox(res$param$n.marker.test)
    )
  )
}

group_data <- json_data$groups
groupNames <- names(group_data)
results <-
  lapply(groupNames, function(k)
    runSkatO(k, group_data[[k]]))

# remove sink for stdout
sink()

# output json with results to stdout
cat(minify(toJSON(
  results, digits = I(4), null = "null", na = "null"
)))