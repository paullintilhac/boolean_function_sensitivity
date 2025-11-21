library(lattice)
library(data.table)
library(readr)
D = 32
F = 8
L = 1
H = 16
MAX_ITERS = 200
path = "~/code/boolean_function_sensitivity/hahn/"
file_string = paste0("d_",D,"-f_",F,"-l_",L,"-h_",H)
newDir = paste0(path,file_string)
f <- data.table(file.info(list.files(newDir, full.names = T)),keep.rownames=T	)
f = f[order(f$mtime)]
f= f[grep(".csv",f$rn)]
recentWide = f[grep("wide",f$rn)][1]$rn
recentLinear = f[grep("linear",f$rn)][1]$rn

getDat = function(filename,max_iters){
  dat =data.table(readr::read_delim(filename, delim = "\t",col_names=F))
  dat = dat[2:nrow(dat),]
  dat = t(dat)
  dat=lapply(dat,function(x) {strsplit(x,",")})
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

if (!is.na(recentLinear)){
  linearDat = getDat(recentLinear,max_iters=MAX_ITERS)
  linearDat$type="linear"
}
if (!is.na(recentWide)){
  wideDat = getDat(recentWide,max_iters=MAX_ITERS)
  wideDat$type = "wide"
}

plotDat = rbind(linearDat,wideDat)

xyplot(value~index|type,groups=variable,plotDat,type="l",main = paste0("Training losses of wide vs linear spectrum functions, Hidden_D=",D,", FF_D=",F,", Layers=",L,", Heads=",H))
