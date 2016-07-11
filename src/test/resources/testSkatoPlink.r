#! /usr/bin/env Rscript

# Setup sink
sink('stdout', type=c("output", "message"))

# load libraries
suppressWarnings(library("SKAT"))
suppressPackageStartupMessages(library("optparse"))

option_list <- list(
  make_option("--plink-root",action="store", help="file path where plink binary files can be found"),
  make_option("--covariates",action="store", help="file path to covariates file (no FID or IID columns)"),
  make_option("--setid", action="store", help="file path to set id file"),
  make_option("--y-type",action="store", help="C or D for y variable type"),
  make_option("--ssd-file", action="store", help="output file for SSD"),
  make_option("--info-file", action="store", help="output file for info file"),
  make_option("--results-file", action="store", help="output file for results"),
  make_option("--ncovar", action="store", type="integer", help="number of covariates"),
  make_option("--seed", action="store", type="integer", default=1, help="random seed"),
  make_option("--n-resampling", action="store", type="integer", default=0, help="number of times to resample residuals"),
  make_option("--type-resampling", action="store", default="bootstrap", help="resampling method"),
  make_option("--no-adjustment", action="store_true", default=FALSE, help="No adjustment for small sample sizes"),
  make_option("--kernel", action="store", default="linear.weighted"),
  make_option("--impute-method", action="store", default="fixed"),
  make_option("--method", action="store", default="davies"),
  make_option("--r-corr", action="store", default="0"),
  make_option("--estimate-maf", action="store", default="1"),
  make_option("--missing-cutoff", action="store", type="double"),
  make_option("--weights-beta", action="store", default="1,25")
)

opt <- parse_args(OptionParser(option_list=option_list))

plinkRoot <- opt$`plink-root`
covar <- opt$covariates
setID <- opt$setid
phenoType <- opt$`y-type`
ssdFile <- opt$`ssd-file`
infoFile <- opt$`info-file`
resultsFile <- opt$`results-file`
nCovar <- opt$ncovar
seed <- opt$seed
rCorr <- as.numeric(strsplit(opt$`r-corr`, ",")[[1]])
weightsBeta <- as.numeric(strsplit(opt$`weights-beta`,",")[[1]])

set.seed(seed)

Generate_SSD_SetID(paste(plinkRoot, ".bed", sep=""), 
                   paste(plinkRoot, ".bim", sep=""),
                   paste(plinkRoot, ".fam", sep=""),
                   setID,
                   ssdFile,
                   infoFile)

SSD.INFO <- Open_SSD(ssdFile, infoFile)

isBinary <- TRUE
if (phenoType == "C") {
  isBinary <-FALSE
}

if (nCovar > 0) {
  PHENO <- suppressWarnings(Read_Plink_FAM(paste(plinkRoot, ".fam", sep=""), Is.binary=isBinary))
  Y <- PHENO$Phenotype
  X <- as.matrix(read.table(covar, header=FALSE))
  for(i in 1:nCovar){
    id.missing <- which(X[,i] == -9)
    if (length(id.missing) > 0){
      X[id.missing,i] <- NA
    }
  }
  obj <- suppressWarnings(SKAT_Null_Model(Y ~ X, out_type=phenoType, n.Resampling = opt$`n-resampling`, type.Resampling = opt$`type-resampling`, Adjustment = !opt$`no-adjustment`))
} else {
  PHENO <- Read_Plink_FAM(paste(plinkRoot, ".fam", sep=""), Is.binary=isBinary)
  Y <- PHENO$Phenotype  
  obj <- suppressWarnings(SKAT_Null_Model(Y ~ 1, out_type=phenoType, n.Resampling = opt$`n-resampling`, type.Resampling = opt$`type-resampling`, Adjustment = !opt$`no-adjustment`))
}

out <- suppressWarnings(SKAT.SSD.All(SSD.INFO, obj, impute.method=opt$`impute-method`, kernel=opt$kernel, method=opt$method, r.corr=rCorr, 
                                     estimate_MAF=opt$`estimate-maf`, missing_cutoff=opt$`missing-cutoff`, weights.beta=weightsBeta))
suppressWarnings(Close_SSD())
sink()
write.table(out$results, resultsFile, row.names = FALSE, col.names = FALSE, quote = FALSE)