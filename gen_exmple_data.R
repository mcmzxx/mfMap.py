rm(list = ls())
basedir='/storage/zhang/mfMap.py/'
setwd(basedir)
oo=c("data", "logs", "results", "ssd", "table") 

organs=c("COADREAD")
for(x in oo){
  if(dir.exists(x))
    unlink(x,recursive=T)
  for(y in organs){
    dir.create(file.path(basedir,x,y),recursive = T)
  }
}
gen_data=function(){
  n=1000;f=500
  sample_name=paste0('s_',as.character(1:n))
  gene_name=paste0('g_',as.character(1:f))
  dna_name=paste0('d_',as.character(1:f))
  features_v1=data.frame(matrix(qnorm(runif(n*f,min=pnorm(0),max=pnorm(1))),f,n))
  features_v2=data.frame(matrix(qnorm(runif(n*f,min=pnorm(0),max=pnorm(1))),f,n))
  rownames(features_v1)=gene_name
  colnames(features_v1)=sample_name
  rownames(features_v2)=dna_name
  colnames(features_v2)=sample_name
  cms=c('CMS1', 'CMS2', 'CMS3', 'CMS4','NOLBL')
  labels=data.frame(barcode=sample_name,subtype=sample(cms,n,replace = T))
  labels$type='tumor'
  labels[which(labels$subtype=='NOLBL'),]$type='cell'
  rownames(labels)=labels$barcode
  write.table(features_v1,file = 'data/COADREAD/features_exp.txt',quote = FALSE,sep = '\t',row.names = T,col.names = T)
  write.table(features_v1,file = 'data/COADREAD/features_mut_cnv_comb.txt',quote = FALSE,sep = '\t',row.names = T,col.names = T)
  write.table(labels,file = 'data/COADREAD/dataset_labels.txt',quote = FALSE,sep = '\t',row.names = T,col.names = T)
}
gen_data()