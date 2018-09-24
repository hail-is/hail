#----------------------------------------------
# Filename: Plot_gq20bydp.R
# Author: Andrea Ganna
# Purpose: Plot GQ >= 20 vs DP bins
#-----------------------------------------------
# Data used: {studyname}_split.gqbydp.tsv
# Data created: {studyname}_split.gqbydp.pdf
#-----------------------------------------------
# OP: R 3.2.0; ggplot2 1.0.1
#-----------------------------------------------*/

## Load libraries
library(ggplot2)
library(reshape)

#filename <- "G77318RH_PASS_NLC_split.gqbydp.tsv"

args <- commandArgs(trailingOnly = TRUE)
filename = as.character(args[1])

Plot_gq20bydp <- function(filename, plotname=".png", widthres=8, heightres=6, title="% of variants with GQ >= 20", xlab="% of variants with GQ >= 20", ylab="DP Bins")
{
	# Read header and data
	headerd <- read.table(filename, nrows = 1, header = FALSE, sep="\t", stringsAsFactors = FALSE)
	d <- read.table(filename, header=F, stringsAsFactor=F, sep="\t", skip=2)
	colnames(d) <- unlist(headerd)

	prefix <- paste(head(strsplit(filename,"\\.")[[1]],n=-1), collapse=".")

	# Create the data for plotting
	meltd <- melt(d)
	meanM <- colMeans(d[,2:ncol(d)],na.rm=T)
	meanMadd <- cbind("Means",names(meanM),colMeans(d[,2:ncol(d)],na.rm=T))
	colnames(meanMadd) <- c("sample","variable","value")
	meltd <- rbind(meltd,meanMadd)

	meltd$ymin <- as.numeric(sapply(strsplit(as.character(meltd$variable),"-"),function(x) x[1]))
	meltd$ymax <- as.numeric(sapply(strsplit(as.character(meltd$variable),"-"),function(x) x[2]))
	meltd$y <- ((meltd$ymax-meltd$ymin)/2)+meltd$ymin

	meltd$sizepoint <- ifelse(meltd$sample == "Means",8,0.001)
	meltd$y <-  ifelse(meltd$sample == "Means",meltd$y,NA)
	meltd$value <-  as.numeric(meltd$value)

	# Plot
	p1 <- ggplot(aes(x=value, ymin = ymin, ymax=ymax, group = sample), data=meltd) + geom_linerange(lwd=0.3, aplha=0.1) + geom_point(aes(y=y,size=sizepoint), pch=21, fill="blue") + ylab(ylab)  + xlab(xlab) + ggtitle(title) + theme(legend.text = element_text(size = 12)) + coord_flip() + scale_y_continuous(ylab,breaks=unique(meltd$y)[!is.na(unique(meltd$y))] ,labels=colnames(d)[2:length(colnames(d))]) + scale_size_continuous(guide=FALSE) + theme_bw()

	ggsave(file=paste0(prefix,plotname),width = widthres, height = heightres, units = 'in')
}


Plot_gq20bydp(filename)




