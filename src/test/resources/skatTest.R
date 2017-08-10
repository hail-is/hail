# SKAT comparison Test

suppressPackageStartupMessages(library(SKAT))

# get temp directory file locations
args <- commandArgs(trailingOnly = TRUE)

G <- read.table(args[1],header =FALSE)
cols = length(G[1,])
G = matrix(unlist(G),ncol = cols,byrow = FALSE)

Cov <- read.table(args[2],header =FALSE)
cols = length(Cov[1,])

Cov = matrix(unlist(Cov),ncol = cols,byrow = FALSE)


#Read in and convert to vectors
pheno <- suppressWarnings(read.table(args[3]))
W <- suppressWarnings(read.table(args[4]))
#This algo takes in sqrt(W) not just simply W
W = unlist(sqrt(W))
pheno = unlist(pheno)

#run SKAT
start.time = Sys.time()
obj <-SKAT_Null_Model(pheno ~ Cov - 1,out_type = args[6])
end.time = Sys.time()
nullModelTime = end.time - start.time
#print(sprintf("Null Model time: %f",nullModelTime))

#print(G)
#print(G %*% diag(sqrt(W)))

#start.time = Sys.time()
sol <- suppressWarnings(SKAT(G,obj,kernel = "linear.weighted",weights = W))
#skatStatTime  = Sys.time() - start.time
#print(sprintf("SKAT time: %f",skatStatTime))

print(sprintf("VCS:     %f",sol$Q[1]))
print(sprintf("p value: %f",sol[1]))

output = c(sol$Q[1],sol[1]$p.value)
#print(output)
#save(list = ls(all.names = TRUE),file = sprintf("/Users/charlesc/Documents/Software/R/workspaces/skatTest%f.RData",runif(1,2.0,5.0)))
write(output,args[5],ncolumns = 2,sep = " ")