library(data.table)
library(lattice)

######
## plots for sharpness of hardcoded vs learned construction
###################
#sharpnessDat = data.table(read.csv("~/Downloads/hardcoded_hessian (11).csv"))
#sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA/hardcoded_hessian.csv"))
#sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2/hardcoded_hessian.csv"))
#sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_NOSAM/hardcoded_hessian.csv"))
#sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_HARDCODED_PERTURBED_001/hardcoded_hessian.csv"))
getHardcodedStats = function(perturbation=0){

#if (perturbation==0) sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_HARDCODED_UNPERTURBED_FULL/hardcoded_hessian.csv"))
if (perturbation==0) sharpnessDat = data.table(read.csv("~/downloads/hardcoded_hessian (14).csv"))
if (perturbation==.001) sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_HARDCODED_PERTURBED_001_FULL/hardcoded_hessian.csv"))
if (perturbation==.0001) sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_HARDCODED_PERTURBED_0001_FULL/hardcoded_hessian.csv"))
if (perturbation==.00001) sharpnessDat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_HARDCODED_PERTURBED_00001_FULL/hardcoded_hessian.csv"))
 
#orig = sharpnessDat[const_mode=="(0, 'original')"] 

#soft = sharpnessDat[grep("original",sharpnessDat$const_mode)]
#balanced = sharpnessDat[const_mode=="(2, 'balanced')"]
plotDat = sharpnessDat[,list(
  "sharpness"=mean(trace_train),
  "norm"=mean(frobenius_weight_norm),
  "sd_sharpness"=sd(trace_train)
  ),by = c("deg","width")]
# plotDat = sharpnessDat
plotDat = plotDat[order(plotDat$deg)]
plotDat = plotDat[deg<=5]
# xyplot(test_sharpness~deg,groups=width,plotDat,type="o",auto.key = TRUE,
#        main = "Trace of Hessian for Exact Construction",
#        xlab = "Degree",
#        ylab = "Trace of Hessian on Training Dataset")
# 
# xyplot(norm~deg,groups=width,plotDat,type="o",auto.key = TRUE,
#        main = "Norm for Exact Construction",
#        xlab = "Degree",
#        ylab = "Trace of Hessian on Training Dataset")
hardPlotDat = plotDat
hardPlotDat$perturbation = perturbation
hardPlotDat$type = "Theoretical"
#return(hardPlotDat)
return(hardPlotDat)

}

getEmpiricalStats = function(SAM = TRUE){
  

if (SAM) { 
  test_dat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2/summary.csv"))
} else {
  #test_dat = data.table(read.csv("~/Downloads/summary (66).csv")) # this is the long run up to deg 4 with dim=2
  # test_dat = rbind(test_dat,data.table(read.csv("~/NEURIPS_CAMERA_NOSAM_FINAL7/summary.csv"))) # this is the long run up to deg 4
  
  #test_dat = rbind(test_dat,data.table(read.csv("~/NEURIPS_CAMERA_NOSAM_FINAL8/summary.csv")))
  
  #test_dat = data.table(read.csv("~/Downloads/summary (70).csv")) # this has up to deg 5 with dim=30
  #test_dat = data.table(read.csv("~/NEURIPS_CAMERA_NOSAM_FINAL10/summary.csv")) 
  
  test_dat = data.table(read.csv("~/Downloads/summary (75).csv")) # this has up to deg 5 with dim=30
  test_dat = rbind(test_dat,data.table(read.csv("~/downloads/summary (73).csv")),fill=TRUE)

  #test_dat = test_dat[n_samples == 16384 & batch_size==1024]
  #test_dat = data.table(read.csv("~/code/boolean_function_sensitivity/NEURIPS_CAMERA2_NOSAM/summary.csv"))
}

#test_dat = data.table(read.csv("~/Downloads/summary (58).csv"))

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
final_dat_single$converged = ifelse(final_dat_single$train_loss<=threshold+.005,TRUE,FALSE)

additionalCols = c("train_loss","gen_gap","time_elapsed","epoch","ln","trace_train","top_eig","weight_norm","converged")
#additionalCols = c("train_loss","gen_gap","time_elapsed","epoch","ln","top_eig","weight_norm","converged")

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

plotDat = plotDat[,list("mean_gen_gap"= mean(gen_gap),
                        "mean_weight_norm" = mean(weight_norm),
                        "func_count" =.N,
                        "mean_hessian_topeig"=mean(top_eig),
                        "mean_hessian_trace"=median(trace_train)),by=split_factors_wo_func]

plotDat$SAM = SAM
plotDat = plotDat[,c("deg","width","mean_hessian_trace","mean_weight_norm","mean_gen_gap","func_count")]
setnames(plotDat,c("deg","width","sharpness","norm","error","count"))


if (SAM) {
  plotDat$type = "Empirical (SAM)"
} else{
  plotDat$type = "Empirical"
}
plotDat = plotDat[order(deg)]

return(plotDat)
}

hardCodedPlot = getHardcodedStats()
hardPlot001 = getHardcodedStats(perturbation = .001)
hardPlot0001 = getHardcodedStats(perturbation = .0001 )
hardPlot00001 = getHardcodedStats(perturbation = .00001 )

empPlot = getEmpiricalStats(SAM = FALSE)
library(ggplot2)
library(patchwork)

# assuming your data is called df and has: sharpness, error, deg, width
# and width is your grouping variable
empPlot$width <- factor(empPlot$width)

p_sharp <- ggplot(empPlot, aes(x = deg, y = sharpness, color = width, group = width)) +
  geom_line() +
  labs(title = "Sharpness vs Degree",
       x = "Degree",
       y = "Sharpness",
       color = "Width") +
  theme_minimal()

p_err <- ggplot(empPlot, aes(x = deg, y = error, color = width, group = width)) +
  geom_line() +
  labs(title = "Error vs Degree",
       x = "Degree",
       y = "Error",
       color = "Width") +
  theme_minimal()


p_sharp + p_err   # side-by-side by default
# or explicitly:
p_sharp | p_err

empPlotSAM = getEmpiricalStats(SAM = TRUE)

combPlot1 = rbind(rbind(rbind(hardCodedPlot,hardPlot001,fill=TRUE), hardPlot0001,fill=TRUE),hardPlot00001,fill=TRUE)
combPlot2 = rbind(hardCodedPlot,empPlot,fill=TRUE)
empPlot = empPlot[order(empPlot$deg)]
combPlot3 =rbind(empPlot,empPlotSAM,fill=TRUE)
combPlot1$sparseString = factor(paste0(combPlot1$width,"-sparse"),levels=c("1-sparse","7-sparse","14-sparse","20-sparse"))
combPlot2$sparseString = factor(paste0(combPlot2$width,"-sparse"),levels=c("1-sparse","7-sparse","14-sparse","20-sparse"))
combPlot3$sparseString = factor(paste0(combPlot3$width,"-sparse"),levels=c("1-sparse","7-sparse","14-sparse","20-sparse"))

#plot just the theoretical sharpness
x

# Optional: You can filter the data within the ggplot call using a pipe operator
# or by creating a subsetted dataframe beforehand.
combPlot2_filtered <- subset(combPlot2, deg <= 5)

ggplot(data = combPlot2_filtered, 
       aes(x = deg, 
           y = log10(sharpness), 
           color = sparseString,   # Map color/grouping to sparseString
           group = sparseString)) +
  
  # Plot the lines (equivalent to type="l")
  geom_line() +
  
  # Facet by 'type' (equivalent to | type)
  facet_wrap(~ type) +
  
  # Add labels and titles (equivalent to ylab, xlab, main, auto.key)
  labs(title = "Comparison of Hessian Trace for Empirical vs Theoretical Bound",
       y = "Trace Hessian",
       x = "Degree",
       color = "Sparsity") + 
  
  # Use a clean theme
  theme_minimal()

# Use the same filtered data frame from above
# combPlot2_filtered <- subset(combPlot2, deg <= 4)

ggplot(data = combPlot2_filtered, 
       aes(x = deg, 
           y = norm, 
           color = sparseString,   # Map color/grouping to sparseString
           group = sparseString)) +
  
  # Plot the lines (equivalent to type="l")
  geom_line() +
  
  # Facet by 'type' (equivalent to | type)
  facet_wrap(~ type) +
  
  # Add labels and titles (equivalent to ylab, xlab, main, auto.key)
  labs(title = "Comparison of Norm for Empirical vs Theoretical Bound",
       y = "Frobenius Norm",
       x = "Degree",
       color = "Sparsity") + 
  
  # Use a clean theme
  theme_minimal()
combPlot1$pert_string = paste0("\u03C3 = ",combPlot1$perturbation)
combPlot1[perturbation==0]$pert_string = "\u03C3 = 0"
combPlot1$pert_string = factor(combPlot1$pert_string,levels=c("σ = 0","σ = 1e-05","σ = 1e-04","σ = 0.001"))


# Ensure upper and lower bounds are calculated:
combPlot1$upper <- combPlot1$sharpness + 1 * combPlot1$sd_sharpness
combPlot1$lower <- combPlot1$sharpness - 1 * combPlot1$sd_sharpness
combPlot1 = combPlot1[deg<=4]
# Define a robust panel function as a closure that can access combPlot1 globally
my_panel_function <- function(x, y, subscripts, group.number, ...) { 
  
  # 1. Get the assigned color for this specific group number using trellis.par.get()
  #    This bypasses the argument passing error entirely.
  group_cols <- trellis.par.get("superpose.line")$col
  current_col <- group_cols[group.number]
  
  # 2. Pass this robustly found color to the panel.xyplot
  panel.xyplot(x, y, col = current_col, ...) 
  
  # 3. Use 'subscripts' (which you must add to the function signature) 
  #    to fetch the correct lower/upper bounds safely:
  current_lower <- combPlot1$lower[subscripts]
  current_upper <- combPlot1$upper[subscripts]
  
  # 4. Use the robust color for the error bars
  panel.segments(x0 = x, y0 = current_lower,
                 x1 = x, y1 = current_upper,
                 col = current_col) 
  
  # Add horizontal caps
  panel.segments(x0 = x - 0.1, y0 = current_lower, 
                 x1 = x + 0.1, y1 = current_lower, 
                 col = current_col)
  panel.segments(x0 = x - 0.1, y0 = current_upper, 
                 x1 = x + 0.1, y1 = current_upper, 
                 col = current_col)
}

# Create the plot using ggplot2
ggplot(data = combPlot1, 
       aes(x = deg, 
           y = sharpness, 
           color = sparseString,   # Color maps automatically to groups
           group = sparseString)) +
  
  # Plot the lines and points
  geom_line() +
  geom_point() +
  
  # Add the error bars using the upper and lower columns
  geom_errorbar(aes(ymin = lower, ymax = upper), 
                width = 0.1,    # Controls the width of the cap
                size = 0.5) +   # Controls the thickness of the bar
  
  # Facet the plot based on pert_string (the sigma value)
  facet_wrap(~ pert_string) +
  
  # Add labels and titles
  labs(title = "Plot of Hessian Trace of Construction with Increasing Perturbations",
       y = "Hessian Trace",
       x = "Degree",
       color = "Sparsity") + # Relabels the legend title
  
  # Set the Y-axis limits (as specified in your original code)
  ylim(-100000, 2000000) +
  
  # Apply a clean theme
  theme_minimal()

## basic harcoded hessian plot
empPlot_filtered <- hardCodedPlot[deg <= 4]

empPlot_filtered$sparseString = factor(paste0(empPlot_filtered$width,"-sparse"),levels=c("1-sparse","7-sparse","14-sparse","20-sparse"))

ggplot(data = empPlot_filtered, 
       aes(x = deg, 
           y = sharpness, 
           color = sparseString,   # Map color/grouping to sparseString
           group = sparseString)) +
  
  # Plot the lines (equivalent to type="l")
  geom_line() +
  
  # Facet by 'type' (equivalent to | type)
  facet_wrap(~ type) +
  
  # Add labels and titles (equivalent to ylab, xlab, main, auto.key)
  labs(title = "Comparison of Hessian Trace for Empirical vs Theoretical Bound",
       y = "Trace Hessian",
       x = "Degree",
       color = "Sparsity") + 
  
  # Use a clean theme
  theme_minimal()
