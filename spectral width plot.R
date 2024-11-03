library(data.table)
dat = data.table(read.csv("~/downloads/test (2).csv"))
lapply(dat,class)
library(lattice)
dat$deg_string = paste0("Degree-",dat$deg)

avgLoss = dat[,list("loss"=mean(loss)),by=c("deg_string","width","iter")]
xyplot(loss~iter|factor(deg_string),groups=width,avgLoss,auto.key=TRUE,type="l",ylim=c(0,1000),xlim=c(1000,60000),main = "Convergence of Construction by Spectral Width")


test_dat = data.table(read.csv("~/Downloads/test_summary.csv"))
test_dat$deg_string = paste0("Degree-",test_dat$deg)
plot_dat=test_dat[,list("avg_test_loss"=mean(test_loss)),by=c("deg_string","width","train_iters")]
xyplot(avg_test_loss~train_iters|factor(deg_string),groups=width,plot_dat,type="l",auto.key=T)
