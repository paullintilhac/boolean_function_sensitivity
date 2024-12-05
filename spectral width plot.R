library(data.table)
library(lattice)


dat = data.table(read.csv("~/downloads/test (2).csv"))
lapply(dat,class)
dat$deg_string = paste0("Degree-",dat$deg)

avgLoss = dat[,list("loss"=mean(loss)),by=c("deg_string","width","iter")]
xyplot(loss~iter|factor(deg_string),groups=width,avgLoss[iter>35000],auto.key=TRUE,type="l",ylim=c(0,1),xlim=c(1000,60000),main = "Convergence of Construction by Spectral Width")


test_dat = data.table(read.csv("~/Downloads/summary (8).csv"))
test_dat$val_loss=as.numeric(gsub(")","", substr(test_dat$val_loss,8,length(test_dat$val_loss))))
test_dat$train_loss=as.numeric(gsub(")","", substr(test_dat$train_loss,8,length(test_dat$train_loss))))
test_dat$deg_string = paste0("Degree-",test_dat$deg)
test_dat$gen_gap = test_dat$val_loss-test_dat$train_loss
test_dat[,final_epoch:=max(epoch),by=c("deg_string","width","func")]
final_dat_single = test_dat[epoch==final_epoch]
final_dat_single[,min_loss:=min(train_loss),by=c("deg","width","func")]
final_dat_single = final_dat_single[train_loss==min_loss]
final_dat_single[,dup_index:=seq_len(.N),by=c("deg","width","func")]
final_dat_single = final_dat_single[dup_index==1]
final_dat_single$exp_trace = unlist(lapply(lapply(strsplit(gsub("\\]","",gsub("\\[","",final_dat_single$trace)),","),as.numeric),mean))
plotDat = final_dat_single[train_loss<.02,c("deg","width","func","train_loss","val_loss","gen_gap","top_eig","deg_string","final_epoch")]
xyplot(gen_gap~deg,
       groups=width,
       plotDat,
       type="l",
       auto.key=T,
       main="Plot of Sharpness by Degree, Width (1-Layer, 1 Head)")

test_dat = data.table(read.csv("~/Downloads/summary (9).csv"))
test_dat$val_loss=as.numeric(gsub(")","", substr(test_dat$val_loss,8,length(test_dat$val_loss))))
test_dat$train_loss=as.numeric(gsub(")","", substr(test_dat$train_loss,8,length(test_dat$train_loss))))
test_dat$deg_string = paste0("Degree-",test_dat$deg)
test_dat$gen_gap = test_dat$val_loss-test_dat$train_loss
test_dat[,final_epoch:=max(epoch),by=c("deg_string","width","func")]
final_dat_double = test_dat[epoch==final_epoch]
final_dat_double = final_dat_double[order(deg)]
final_dat_double[,min_loss:=min(train_loss),by=c("deg","width","func")]
final_dat_double = final_dat_double[train_loss==min_loss]
final_dat_double[,dup_index:=seq_len(.N),by=c("deg","width","func")]
final_dat_double = final_dat_double[dup_index==1]
final_dat_double$exp_trace = unlist(lapply(lapply(strsplit(gsub("\\]","",gsub("\\[","",final_dat_double$trace)),","),as.numeric),mean))

final_dat_single$deg=as.numeric(final_dat_single$deg)
final_dat_double$deg=as.numeric(final_dat_double$deg)

cbind(lapply(final_dat_single,class),lapply(final_dat_double,class))
final_dat = rbind(final_dat_single,final_dat_double)
final_dat$width_string = paste0("width-",final_dat$width)
final_dat$layer_string =paste0(final_dat$l,"-layer")
plot_dat = final_dat[,list("mean_norm"=mean(top_eig),"mean_trace"=mean(exp_trace),"mean_gen_gap"=mean(gen_gap)),by=c("deg","width_string","layer_string")]
xyplot(mean_gen_gap~deg|factor(layer_string),
       groups=width_string,
       plot_dat[deg<5],
       auto.key=T,
       type="o",
       main = "gen gap by width, degree, #layers")

xyplot(mean_trace~deg|factor(layer_string),
       groups=width_string,
       plot_dat[deg<5],
       auto.key=T,
       type="o",
       main = "mean hessian trace by width, degree, #layers")

xyplot(mean_norm~deg|factor(layer_string),
       groups=width_string,
       plot_dat[deg<5],
       auto.key=T,
       type="o",
       main = "mean hessian norm by width, degree, #layers")

xyplot(gen_gap~width,
       groups=deg_string,
       final_dat_single,
       auto.key = T,
       type="l",
       main = "generalization gap by width and degree")
write.csv(final_dat, file = "hello.csv")
#test_dat[,min(gen_gap),by=c("deg_string","width","func")]
#plot_dat=test_dat[,list("avg_test_loss"=mean(test_loss)),by=c("deg_string","width","epoch")]
xyplot(gen_gap~epoch|factor(deg_string),
       groups=width,
       test_dat,
       ylim=c(0,1),
       type="l",
       auto.key=T)

