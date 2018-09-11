#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

t <- args[1]
a <- as.integer(args[2])
b <- as.integer(args[3])
c <- as.integer(args[4])
d <- as.integer(args[5])

fet <- fisher.test(matrix(c(a,b,c,d),2,2), alternative = t)
cat(paste(c(fet$p.value, fet$estimate, paste(fet$conf.int, sep=" "), sep=" ")))