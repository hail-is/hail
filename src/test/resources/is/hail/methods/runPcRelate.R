args <- commandArgs(trailingOnly = TRUE)

library(GENESIS)
library(SNPRelate)
library(GWASTools)

fname = args[1]

snpgdsBED2GDS( bed.fn = paste(fname,".bed",sep="")
             , bim.fn = paste(fname,".bim",sep="")
             , fam.fn = paste(fname,".fam",sep="")
             , out.gdsfn = paste(fname,".gds",sep="")
             )

gdsfile <- system.file("extdata", paste(fname,".gds",sep=""), package="GENESIS")
HapMap_geno <- GdsGenotypeReader(filename = paste(fname,".gds",sep=""))
HapMap_genoData <- GenotypeData(HapMap_geno)

mypcair <- pcair(genoData = HapMap_genoData)

options(digits=18)
print(mypcair$vectors[,1:2])
mypcrelate <- pcrelate( genoData = HapMap_genoData
                      , pcMat = mypcair$vectors[,1:2]
                      , correct = FALSE
                      , MAF = 0.0
                      )
foo <- pcrelateReadKinship( pcrelObj = mypcrelate
                          )

write.table(foo, paste(fname,".out",sep=""), row.names=FALSE, quote=FALSE)
