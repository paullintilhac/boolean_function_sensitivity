library(lattice)
library(data.table)
library(readr)
D = 64
F = 4
L = 1
H = 2
MAX_ITERS = 300
path = "~/code/boolean_function_sensitivity/hahn/"
file_string = paste0("d_",D,"-f_",F,"-l_",L,"-h_",H)
newDir = paste0(path,file_string)
f <- data.table(file.info(list.files(newDir, full.names = T)),keep.rownames=T	)
f = f[order(f$mtime,decreasing=T)]
f= f[order(f$size,decreasing=T)]
f= f[grep(".csv",f$rn)]
recentWide = f[grep("wide",f$rn)][1]$rn
recentLinear = f[grep("linear_spectrum_small_[0-9]",f$rn)][1]$rn
recentNoOverlap=f[grep("linear_spectrum_small_no",f$rn)][1]$rn


getDat = function(filename,max_iters){
  print(filename)
  dat =data.table(readr::read_delim(filename, delim = "\t",col_names=F))
  if (nchar(dat[1]$X1)<100){
    dat = dat[2:nrow(dat),]
  }
  
  dat = t(dat)
  tmp = dat
  
  if (length(lapply(dat,function(x) {strsplit(x,",")})[[1]][[1]])>1){
    dat=lapply(dat,function(x) {strsplit(x,",")})
  } 
  
  if (length(dat[[1]][[1]])>1){
  if (length(lapply(tmp,function(x) {strsplit(x,"\t")})[[1]][[1]])>1){
    dat=lapply(dat,function(x) {strsplit(x,"\t")})
  }}
  numRecs = length(dat[[1]][[1]])
  mat = matrix(0,numRecs,length(dat))
  for (i in 1:length(dat)){
    mat[,i]=as.numeric(dat[[i]][[1]])
  }
  mat = data.table(mat)
  mat$index = seq(1,numRecs)
  finaldat = melt.data.table(data.table(mat),id.vars = "index")
  finaldat = finaldat[index<=max_iters,]
}

linearDat = getDat(recentLinear,max_iters=MAX_ITERS)
linearDat$type="linear"
wideDat = getDat(recentWide,max_iters=MAX_ITERS)
wideDat$type = "wide"

noOverlapDat = getDat(recentNoOverlap,max_iters = MAX_ITERS)
noOverlapDat$type= "linear_no_overlap"
plotDat = rbind(linearDat,wideDat)
plotDat = rbind(linearDat,wideDat)
plotDat = rbind(rbind(linearDat,wideDat),noOverlapDat)

xyplot(value~index|type,groups=variable,plotDat,type="l",main = paste0("Training losses of wide vs linear spectrum functions, Hidden_D=",D,", FF_D=",F,", Layers=",L,", Heads=",H))

