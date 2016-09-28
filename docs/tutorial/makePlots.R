#! /usr/bin/env Rscript

sampleqc <- read.table("test.sampleqc.tsv", header=TRUE)
variantqc <- read.table("test.variantqc.tsv", header=TRUE)
linreg <- na.omit(read.table("test.linreg.tsv", header=TRUE))
logreg <- na.omit(read.table("test.logreg.tsv", header=TRUE))
pcs <- read.table("test.pcaPlusPop.tsv", header=TRUE)
fet <- na.omit(read.table("test.fet.tsv", header=TRUE))

## Sample QC
png("test.sampleqc.png", width=800, height = 400)
par(mfrow=c(1,2))
hist(sampleqc$callRate, breaks=20, xlab="Sample Call Rate", main=""); abline(v=0.97, col="red")
hist(sampleqc$gqMean, xlab="Mean Sample GQ", main="");abline(v=20, col="red")
dev.off()

## Variant QC
png("test.variantqc.png", width=800, height = 400)
par(mfrow=c(2,2))
hist(variantqc$gqMean, main="", xlab="Variant Mean GQ"); abline(v=20, col="red")
hist(variantqc$AF, main="", xlab="Minor Allele Frequency", breaks=20); abline(v=0.05, col="red")
hist(variantqc$callRate, main="", xlab="Variant Call Rate")
hist(variantqc$pHWE, main="", xlab="Hardy-Weinberg Equilibrium p-value")
dev.off()

## PC Plot
png("test.pcaPlot.png", width=800, height = 400)
par(mfrow=c(1,2))
plot(pcs$PC1, pcs$PC2, pch=20, cex=0.7, col=pcs$SuperPopulation, xlab="PC1", ylab="PC2")
legend("topright", legend = levels(pcs$SuperPopulation), pch=20, col=seq(1,5))
plot(pcs$PC2, pcs$PC3, pch=20, cex=0.7, col=pcs$SuperPopulation, xlab="PC2", ylab="PC3")
legend("topright", legend = levels(pcs$SuperPopulation), pch=20, col=seq(1,5))
dev.off()

## Linear Regression
png("test.linreg.qq.png")
obs <- -1*log(sort(linreg$PVAL), 10)
exp <- -log(seq(1, length(linreg$PVAL)) / length(linreg$PVAL), 10)
plot(exp, obs, pch=20, cex=1); abline(a=0, b=1, col="red", lwd=2)
dev.off()

## Logistic Regression
png("test.logreg.qq.png")
obs <- -1*log(sort(logreg$PVAL), 10)
exp <- -log(seq(1, length(logreg$PVAL)) / length(logreg$PVAL), 10)
plot(exp, obs, pch=20, cex=1); abline(a=0, b=1, col="red", lwd=2)
dev.off()

## Fisher Exact Test
png("test.fet.qq.png")
obs <- -1*log(sort(fet$PVAL), 10)
exp <- -log(seq(1, length(fet$PVAL)) / length(fet$PVAL), 10)
plot(exp, obs, pch=20, cex=1); abline(a=0, b=1, col="red", lwd=2)
dev.off()