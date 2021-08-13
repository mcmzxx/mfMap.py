rm(list = ls())
basedir='/storage/zhang/mfMap.py/'
setwd(basedir)
oo=c("data", "logs", "results", "ssd", "table") 

organs=c("COADREAD")
for(x in oo){
  if(dir.exists(x))
    unlink(x,recursive=T)
}