#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

t <- args[1]
a <- as.integer(args[2])
b <- as.integer(args[3])
c <- as.integer(args[4])
d <- as.integer(args[5])

cat(fisher.test(matrix(c(a,b,c,d),2,2), alternative = t)$p.value)
