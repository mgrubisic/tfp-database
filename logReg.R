# Huy Pham
# UC Berkeley
# Created: October 2020
# Description: This script does logistic regression to determine param importance

library(tidyverse)

rm(list=ls())

# dataPath <- '../pastRuns/random200withTfb.csv'
# 
# isolDat <- read.csv(dataPath, header=TRUE)
# 
# # non dims
# isolDat$TfbRatio <- isolDat$Tfb/isolDat$Tm
# isolDat$mu2Ratio <- isolDat$mu2/isolDat$GMSTm
# isolDat$gapRatio <- isolDat$moatGap/(isolDat$GMSTm*isolDat$Tm^2)
# isolDat$T2Ratio <- isolDat$T2/isolDat$Tm
# isolDat$Ry <- isolDat$RI
# isolDat$zeta <- isolDat$zetaM
# isolDat$A_S1 <- isolDat$S1Ampli
# 
# isolDat$collapsed <- as.numeric(isolDat$collapseDrift1 |
#                                   isolDat$collapseDrift2 |
#                                   isolDat$collapseDrift3)
# 
# collapsedLogit <- glm(collapsed ~ TfbRatio + mu2Ratio + gapRatio + T2Ratio +
#                         Ry + zeta, data = isolDat)
# 
# summary(collapsedLogit)
# 
# impactedLogit <- glm(impacted ~ TfbRatio + mu2Ratio + gapRatio + T2Ratio +
#                         Ry + zeta, data = isolDat)
# 
# summary(impactedLogit)
# 
# piGroups <- data.frame(isolDat$TfbRatio,
#                        isolDat$mu2Ratio,
#                        isolDat$gapRatio,
#                        isolDat$T2Ratio,
#                        isolDat$Ry,
#                        isolDat$zeta)
# 
# corMatPi <- piGroups %>% as.matrix %>% cor %>% as.data.frame
# 
# rawVars <- data.frame(isolDat$Ry,
#                       isolDat$Tfb,
#                       isolDat$Tm,
#                       isolDat$mu2,
#                       isolDat$moatGap,
#                       isolDat$zetaM,
#                       isolDat$T2,
#                       isolDat$GMSTm)
# 
# corMatRaw <- rawVars %>% as.matrix %>% cor %>% as.data.frame
# 

collapsed <- c(2, 25, 43)
pCol <- collapsed/54
imLevel <- c(1, 1.5, 2)

# collapsed <- c(1, 9, 21, 30, 38, 43)
# pCol <- collapsed/54
# imLevel <- seq(1, 2.25, by = 0.25)

# fit lognormal parameters using binomial likelihood function
startPt <- c(mean(log(imLevel)), sqrt(var(log(imLevel))))

negBinomLik <- function(lnormPars){
  p <- pnorm(log(imLevel), mean = lnormPars[1], sd = lnormPars[2])
  binomPdfs <- dbinom(collapsed, size = 54, prob = p)
  likelihood <- prod(binomPdfs)
  return(-likelihood)
}

fit <- optim(startPt, negBinomLik)

theta <- exp(fit$par[1])
beta <- fit$par[2]

ims <- seq(0, 3, length.out = 100)
fitcdf <- pnorm(ims, mean = theta, sd = beta)

idaFit <- data.frame(ims, fitcdf)
idaData <- data.frame(imLevel, collapsed, pCol)

ggplot(data = idaFit, aes(ims, fitcdf)) + geom_line() +
  geom_point(data = idaData, aes(imLevel, pCol))

# ida <- data.frame(imLevel, pCol)
# 
# colFit <- glm(pCol ~ imLevel, data = ida, family = "binomial")
# 
# idaNew <- data.frame(imLevel = seq(0.8, 3.0, length.out = 100))
# idaNew$probs <- predict(colFit, newdata=idaNew, type = "response")
