library(data.table)
dat = data.table(read.csv("~/downloads/test (2).csv"))
lapply(dat,class)
library(lattice)
dat$deg_string = paste0("Degree-",dat$deg))

avgLoss = dat[,list("loss"=mean(loss)),by=c("deg","width","iter")]
xyplot(loss~iter|factor(deg_string),groups=width,avgLoss,auto.key=TRUE,type="l",ylim=c(0,1),xlim=c(20000,60000),main = "Convergence of Construction by Spectral Width")
