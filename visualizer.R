# Huy Pham
# UC Berkeley
# Created: September 2020
# Description: This script plots data from the isolation data

library(ggplot2)
library(dplyr)

dataPath <- '../pastRuns/random200withTfb.csv'

isolDat <- read.csv(dataPath, header=TRUE)
#isolDat <- filter(isolDat, impacted == 1)
isolDat$maxDrift <- pmax(isolDat$driftMax1, isolDat$driftMax2, isolDat$driftMax3)
isolDat$collapse <- (isolDat$collapseDrift1 | isolDat$collapseDrift2) | 
  isolDat$collapseDrift3

g <- 386.4

# nondims
isolDat$TfbRatio <- isolDat$Tfb/isolDat$Tm
isolDat$mu2Ratio <- isolDat$mu2/isolDat$GMSTm
# isolDat$gapRatio <- isolDat$moatGap/(isolDat$mu2 * g * isolDat$Tm^2)
isolDat$gapRatio <- isolDat$moatGap/(isolDat$GMSTm * g * isolDat$Tm^2)
isolDat$T2Ratio <- isolDat$T2/isolDat$Tm
isolDat$Qm <- isolDat$mu2*g

# point plot function
dots <- function(mapping){
  ggplot(data = isolDat, mapping) + 
    geom_point()
}

# mu2 vs T
dots(aes(Qm, Tm, color = T2))
dots(aes(mu2, GMSTm, color = mu2Ratio))
dots(aes(GMST2, GMSTm))
dots(aes(RI, T1, color = Tm))
dots(aes(Tm, moatGap, color = gapRatio))

dots(aes(zetaM, moatGap, color = Tm))
dots(aes(RI, Tfb, color = T2))
dots(aes(Tfb, T2, color = T1))
dots(aes(gapRatio, mu2Ratio))

# variables vs collapse
dots(aes(gapRatio, maxDrift))
dots(aes(T2Ratio, maxDrift))
dots(aes(zetaM, maxDrift))

dots(aes(TfbRatio, T2Ratio))
