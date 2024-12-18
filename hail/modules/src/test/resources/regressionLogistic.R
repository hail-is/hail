suppressPackageStartupMessages(library("jsonlite"))
suppressPackageStartupMessages(library("logistf"))

args <- commandArgs(trailingOnly=TRUE)
#args <- c("/tmp/input.json", "/tmp/wald.tsv", "/tmp/lrt.tsv", "/tmp/score.tsv", "/tmp/firthB.tsv", "/tmp/firthlrt.tsv")

json_data <-
  fromJSON(args[1], simplifyVector = TRUE, simplifyMatrix = TRUE)

nrow <- json_data$rows
ncol <- json_data$cols
y <- json_data$y
X <- matrix(json_data$X, nrow, ncol)

logfit <- glm(y ~ X, family=binomial(link="logit"), control=glm.control(epsilon = 1e-8))
waldcoef <- coef(summary(logfit))

write.table(waldcoef, args[2], row.names=FALSE, col.names=FALSE, sep="\t")

lrChi2 <- logfit[["null.deviance"]] - logfit[["deviance"]]
lrPVal <- 1 - pchisq(lrChi2, 2)

write.table(c(lrChi2, lrPVal), args[3], row.names=FALSE, col.names=FALSE)

scorefit <- anova(logfit, test="Rao")
scoreChi2 <-scorefit[["Rao"]][2]
scorePVal <- scorefit[["Pr(>Chi)"]][2]

write.table(c(scoreChi2, scorePVal), args[4], row.names=FALSE, col.names=FALSE, sep="\t")

firthfit <- logistf(y ~ X, control=logistf.control(xconv=1e-8))
firthcoef <- firthfit["coefficients"]

write.table(firthcoef, args[5], row.names=FALSE, col.names=FALSE, sep="\t")

firthloglik <- firthfit[["loglik"]]
firthpval <- logistftest(firthfit)[["prob"]]

write.table(c(firthloglik, firthpval), args[6], row.names=FALSE, col.names=FALSE, sep="\t")
