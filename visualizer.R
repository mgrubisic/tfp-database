# Huy Pham
# UC Berkeley
# Created: September 2020
# Description: This script plots data from the isolation data

library(ggplot2)

isolDat <- read.csv('./sessionOut/sessionSummary.csv', header=TRUE)
isolDat$maxDrift <- pmax(isolDat$driftMax1, isolDat$driftMax2, isolDat$driftMax3)

isolPlot <- ggplot(isolDat, aes(GMSavg, maxDisplacement, colour = driftMax2, size = R1)) +
  geom_point()

isolPlot

intensityMeasures <- ggplot(isolDat, aes(GMSavg, GMS1, colour = S1, size = moatGap)) +
  geom_point()

intensityMeasures

nonDimPlot <- ggplot(isolDat, aes(Pi3, Pi4, colour = maxDrift, size = maxDisplacement)) +
  geom_point()

nonDimPlot

gapPlot <- ggplot(isolDat, aes(moatAmpli, maxDrift, colour = moatGap, size = maxDisplacement)) +
  geom_point()

gapPlot
