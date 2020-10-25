# Huy Pham
# UC Berkeley
# Created: October 2020
# Description: This script does logistic regression to determine param importance

library(tidyverse)

rm(list=ls())

dataPath <- '../pastRuns/random200withTfb.csv'

isolDat <- read.csv(dataPath, header=TRUE)

# non dims
isolDat$TfbRatio <- isolDat$Tfb/isolDat$Tm
isolDat$mu2Ratio <- isolDat$mu2/isolDat$GMSTm
isolDat$gapRatio <- isolDat$moatGap/(isolDat$GMSTm*isolDat$Tm^2)
isolDat$T2Ratio <- isolDat$GMST2/isolDat$GMSTm
isolDat$Ry <- isolDat$RI
isolDat$zeta <- isolDat$zetaM
isolDat$A_S1 <- isolDat$S1Ampli

isolDat$collapsed <- as.numeric(isolDat$collapseDrift1 | 
                                  isolDat$collapseDrift2 | 
                                  isolDat$collapseDrift3)

collapsedLogit <- glm(collapsed ~ TfbRatio + mu2Ratio + gapRatio + T2Ratio +
                        Ry + zeta + A_S1, data = isolDat)

summary(collapsedLogit)

impactedLogit <- glm(impacted ~ TfbRatio + mu2Ratio + gapRatio + T2Ratio +
                        Ry + zeta + A_S1, data = isolDat)

summary(impactedLogit)

