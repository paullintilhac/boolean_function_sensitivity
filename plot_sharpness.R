library(data.table)
library(lattice)

######
## plots for sharpness of hardcoded vs learned construction
###################
sharpnessDat = data.table(read.csv("/scratch/plintilhac/HESSIAN_CALCS22/hardcoded_hessian.csv"))

#orig = sharpnessDat[const_mode=="(0, 'original')"] 

soft = sharpnessDat[grep("original",sharpnessDat$const_mode)]
#balanced = sharpnessDat[const_mode=="(2, 'balanced')"]
plotDat = soft[,sharpness:=mean(trace_train),by = c("deg","width")]
plotDat = plotDat[order(plotDat$deg)]
plotDat$width_string = paste0("sparsity-",plotDat$width)
plot1 = xyplot(sharpness~deg,groups=width_string,plotDat,type="l",auto.key = TRUE,
       main = "Trace of Hessian for Exact Construction",
       xlab = "Degree",
       ylab = "Trace of Hessian on Training Dataset")

png("hardcoded sharpness plot.png")
print(plot1)
dev.off()

hardPlotDat = plotDat

test_dat = data.table(read.csv("/scratch/plintilhac/HESSIAN_CALCS22/summary.csv"))
#test_dat = data.table(read.csv("~/Downloads/small degs large widths.csv"))
#test_dat = data.table(read.csv("~/Downloads/HYPERPARAM_TESTS_MECHINTERP/summary.csv"))

test_dat[-1,]
test_dat$trace_charlen = nchar(test_dat$trace)
test_dat$trace=unlist(lapply(strsplit(substr(substr(test_dat$trace,2,test_dat$trace_charlen),1,test_dat$trace_charlen-2),","),function(x) { mean(as.numeric(x))}))
test_dat$val_loss=as.numeric(gsub(")","", substr(test_dat$val_loss,8,length(test_dat$val_loss))))
test_dat$train_loss=as.numeric(gsub(")","", substr(test_dat$train_loss,8,length(test_dat$train_loss))))
test_dat$deg_string = paste0("Degree-",test_dat$deg)
test_dat$gen_gap = test_dat$val_loss-test_dat$train_loss

#calculate theoretical generalization gap based on width, degree, T, delta,
# and the free parameter sigma which I chose from a quick coarse-grained 
# optimization /visual inspection of a grid of options. Sigma is calculated from
# upper bounding  the empirical moment generating function of the validation loss
# with a family of gaussian mgfs (the least upper bound)
T=50
  
split_factors = c("deg",
                  "deg_string",
                  "width","func",
                  "batch_size",
                  "lr",
                  "dropout",
                  "wd",
                  "d",
                  "n_samples",
                  "stop_loss",
                  "f")
THRESHOLD=TRUE
if (THRESHOLD){
  threshold = .02
  test_dat = test_dat[train_loss<=threshold]
  test_dat[,final_epoch:=min(epoch),by=split_factors]
} else{
  test_dat[,final_epoch:=max(epoch),by=split_factors]
}

final_dat_single = test_dat[epoch==final_epoch]
final_dat_single[,min_loss:=min(train_loss),by=split_factors]
final_dat_single = final_dat_single[train_loss==min_loss]
final_dat_single[,dup_index:=seq_len(.N),by=split_factors]
final_dat_single = final_dat_single[dup_index==1]
final_dat_single$converged = ifelse(final_dat_single$train_loss<=threshold,TRUE,FALSE)

additionalCols = c("train_loss","gen_gap","time_elapsed","epoch","ln","trace_train","top_eig","weight_norm","converged")
keepCols = c(split_factors,additionalCols)
final_dat_single[,..keepCols]
#final_dat_single[c(11,12,14,15),]
final_dat_single = final_dat_single[converged==TRUE]
#check final_dat_single here to see final info for all unique training runs.#check final_dat_single hdelere to see final info for all unique training runs.

split_factors_wo_func = c("deg","deg_string","width","batch_size","lr","dropout","wd","d","n_samples","stop_loss","f")
#final_dat_single$exp_trace = unlist(lapply(lapply(strsplit(gsub("\\]","",gsub("\\[","",final_dat_single$trace)),","),as.numeric),mean))
#plotDat = final_dat_single[train_loss<.02&test_num==3,c("deg","width","func","train_loss","val_loss","gen_gap","top_eig","deg_string","final_epoch","n_samples","batch_size","lr")]
keepCols2 = c(keepCols,"final_epoch")
plotDat = final_dat_single[,..keepCols2]

plotDat[,mean_gen_gap := mean(gen_gap),by=split_factors_wo_func]
plotDat[,mean_weight_norm := mean(weight_norm),by=split_factors_wo_func]

plotDat[,func_count :=.N,by=split_factors_wo_func]
plotDat[,mean_hessian_topeig:=mean(top_eig),by=split_factors_wo_func]
plotDat[,mean_hessian_trace:=median(trace_train),by=split_factors_wo_func]

plotDat = plotDat[order(deg_string)]
plot2 = xyplot(mean_hessian_trace~deg,
     groups=width,
     plotDat,
     type="l",
     auto.key=TRUE,
     ylab = "Empirical Trace of Hessian on Training Dataset",
     xlab = "Degree",
     main="Plot of Empirical Hessian Trace")
png("empirical sharpness plot.png")
print(plot2)
dev.off()
# 
# plot(plot1, split = c(1, 1, 1, 2)) # Plot 1 at (1,1) in a 1x2 grid
# plot(plot2, split = c(1, 2, 1, 2), newpage = FALSE)
# 
# plotDat$sharpness = plotDat$mean_hessian_trace
# plotDat$mode = "Empirical"
# hardPlotDat$sharpness = hardPlotDat$V1
# hardPlotDat$mode = "Construction"
# stackedDat = rbind(plotDat,hardPlotDat,fill=TRUE)
# stackedDat$width_string = paste0("sparsity-",stackedDat$width)
# 
# xyplot(sharpness~deg|mode,
#        groups=width_string,
#        stackedDat[deg<=3],
#        type="l",
#        auto.key=TRUE,
#        ylab = "Empirical Trace of Hessian on Training Dataset",
#        xlab = "Degree",
#        main="Plot of Empirical Hessian Trace")
# 
# library(latticeExtra)
# doubleYScale(
#   xyplot(mean_hessian_trace~deg,
#          groups=width,
#          plotDat[deg<=3],
#          type="l",
#          auto.key=TRUE,
#          ylab = "Empirical Trace of Hessian on Training Dataset",
#          xlab = "Degree",
#          main="Plot of Empirical Hessian Trace"),
#   xyplot(V1~deg,groups=width,hardPlotDat,type="l",auto.key = TRUE,
#          main = "Trace of Hessian by Deg, Sparsity"),
#   add.ylab2 = TRUE,
#   text.columns = c("Variable 1", "Variable 2"),
#   columns = c("blue", "red")
# )
# finalPlotDat = merge(plotDat,hardPlotDat,by=c("deg","width"))
# finalPlotDat = finalPlotDat[deg<=3]
# 
# plotDat$width_string = paste0("sparsity-",plotDat$width)
# 
# xyplot(V1+mean_hessian_trace~deg,
#        groups=width,
#        finalPlotDat,
#        type="l",
#        auto.key=TRUE,
#        main="Plot of Theoretical Generalization Gap by Degree, Width, \U03BB (low \U03A3)")
# 
# 
# 
# 

