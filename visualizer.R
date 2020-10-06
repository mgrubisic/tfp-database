# Huy Pham
# UC Berkeley
# Created: September 2020
# Description: This script plots data from the isolation data

library(ggplot2)

dataPath <- '../pastRuns/random200withST.csv'
#dataPath <- './sessionOut/sessionSummary.csv'

isolDat <- read.csv(dataPath, header=TRUE)
isolDat$maxDrift <- pmax(isolDat$driftMax1, isolDat$driftMax2, isolDat$driftMax3)

# nondims
isolDat$Pi1 <- isolDat$mu1/isolDat$GMSTm
isolDat$Pi2 <- isolDat$Tm^2/(386.4/isolDat$R1)

isolDat$Pi3 <- isolDat$T2/isolDat$T1
isolDat$Pi4STm <- isolDat$mu2/isolDat$GMSTm
isolDat$Pi4ST1 <- isolDat$mu2/isolDat$GMST1
isolDat$Pi4ST2 <- isolDat$mu2/isolDat$GMST2
isolDat$Pi4S1 <- isolDat$mu2/isolDat$GMS1
isolDat$Pi4Savg <- isolDat$mu2/isolDat$GMSavg

# displacement vs GM
displPlot <- ggplot(isolDat, aes(GMSavg, maxDisplacement, colour = driftMax2, size = R1)) +
  geom_smooth() + geom_point()

displPlot

#intensityMeasures <- ggplot(isolDat, aes(GMSavg, GMS1, colour = S1, size = moatGap)) +
  #geom_point()

#intensityMeasures

# gap overdesign vs drift
gapPlot <- ggplot(isolDat, aes(moatAmpli, maxDrift, size = moatGap,
                               color = impacted)) +
  geom_point() + scale_shape_identity()

gapPlot

# gap overdesign vs impact
bp <- ggplot(data = isolDat, aes(x = as.factor(impacted), y = moatAmpli)) +
  geom_boxplot() +
  xlab("Impacted?") + ylab("Moat amplification")

bp

# structure overstrength
RIDrift <- ggplot(isolDat, aes(RI, maxDrift, colour = impacted, size = moatAmpli)) +
  geom_point()

RIDrift

# bigger EQ than expected

S1Drift <- ggplot(isolDat, aes(S1Ampli, maxDrift, colour = impacted, size = GMSTm)) +
  geom_point()

S1Drift

S1Displ <- ggplot(isolDat, aes(S1Ampli, maxDisplacement, colour = impacted, size = GMS1)) +
  geom_point()

S1Displ

# damping
zDrift <- ggplot(isolDat, aes(zetaM, maxDrift, colour = impacted, size = Tm)) +
  geom_point()

zDrift

# Tm group

TmDrift <- ggplot(isolDat, aes(Pi1, maxDrift, colour = impacted)) +
  geom_point()

TmDrift

TmDrift2 <- ggplot(isolDat, aes(Pi1, maxDisplacement, colour = impacted)) +
  geom_point()

TmDrift2

# T2 group

correlatedPis <- ggplot(isolDat, aes(Pi3, Pi4STm)) +
  geom_point()

correlatedPis

T2Drift <- ggplot(isolDat, aes(Pi3, maxDrift, colour = impacted, size = Pi4STm)) +
  geom_point()

T2Drift

T2Drift2 <- ggplot(isolDat, aes(Pi4STm, maxDrift, colour = impacted)) +
  geom_point()

T2Drift2

# Study of Sa's on displacement
displCorr <- c("STm" = cor(isolDat$maxDisplacement, isolDat$Pi4STm),
               "ST2" = cor(isolDat$maxDisplacement, isolDat$Pi4ST2),
               "ST1" = cor(isolDat$maxDisplacement, isolDat$Pi4ST1),
               "S1" = cor(isolDat$maxDisplacement, isolDat$Pi4S1),
               "Savg" = cor(isolDat$maxDisplacement, isolDat$Pi4Savg))
  
  
STmDispl <- ggplot(isolDat, aes(Pi4STm, maxDisplacement, colour = impacted)) +
  geom_point()

STmDispl

ST1Displ <- ggplot(isolDat, aes(Pi4ST1, maxDisplacement, colour = impacted)) +
  geom_point()

ST1Displ

ST2Displ <- ggplot(isolDat, aes(Pi4ST2, maxDisplacement, colour = impacted)) +
  geom_point()

ST2Displ

S1Displ <- ggplot(isolDat, aes(Pi4S1, maxDisplacement, colour = impacted)) +
  geom_point()

S1Displ

SavgDispl <- ggplot(isolDat, aes(Pi4Savg, maxDisplacement, colour = impacted)) +
  geom_point()

SavgDispl

# Study of Sa's on drift
driftCorr <- c("STm" = cor(isolDat$maxDrift, isolDat$Pi4STm),
               "ST2" = cor(isolDat$maxDrift, isolDat$Pi4ST2),
               "ST1" = cor(isolDat$maxDrift, isolDat$Pi4ST1),
               "S1" = cor(isolDat$maxDrift, isolDat$Pi4S1),
               "Savg" = cor(isolDat$maxDrift, isolDat$Pi4Savg))

STmDrift <- ggplot(isolDat, aes(Pi4STm, maxDrift, colour = impacted)) +
  geom_point()

STmDrift

ST1Drift <- ggplot(isolDat, aes(Pi4ST1, maxDrift, colour = impacted)) +
  geom_point()

ST1Drift

ST2Drift <- ggplot(isolDat, aes(Pi4ST2, maxDrift, colour = impacted)) +
  geom_point()

ST2Drift

S1Drift <- ggplot(isolDat, aes(Pi4S1, maxDrift, colour = impacted)) +
  geom_point()

S1Drift

SavgDrift <- ggplot(isolDat, aes(Pi4Savg, maxDrift, colour = impacted)) +
  geom_point()

SavgDrift
