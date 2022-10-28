###############################################
#########  load packages and clean data
###############################################

library(mgcv)
library(keras)
library(magrittr)
library(deepviz)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(gridExtra)
library(splitTools)
library(tidyr)
library(mltools)
library(factoextra)
library(devtools)

###############################################
#########  Poisson deviance statistics
###############################################

Poisson.Deviance <- function(pred, obs){200*(sum(pred)-sum(obs)+sum(log((obs/pred)^(obs))))/length(pred)}

# data is available from http://www2.math.uconn.edu/~valdez/data.html

# load rawdata
data <- read.csv("telematics_syn-032021.csv")
data$Duration <- pmin(data$Duration/365, 1) # insurance exposure

###############################################
#########  feature pre-processing for GLM
###############################################

datT <- data[,c(1:11,51, 52)]
datT$Insured.sex <- as.factor(datT$Insured.sex)
datT[,"Insured.sex"] <-relevel(datT[,"Insured.sex"], ref="Male")
datT$Marital <- as.factor(datT$Marital)
datT[,"Marital"] <-relevel(datT[,"Marital"], ref="Married")
datT$Car.use <- as.factor(datT$Car.use)
datT[,"Car.use"] <-relevel(datT[,"Car.use"], ref="Private")
datT$Region <- as.factor(datT$Region)
datT[,"Region"] <-relevel(datT[,"Region"], ref="Urban")
datT$Territory <- as.factor(datT$Territory)
datT[,"Territory"] <-relevel(datT[,"Territory"], ref="11")

PreProcess.Continuous <- function(var1, datT){
  names(datT)[names(datT) == var1]  <- "V1"
  datT$X <- as.numeric(datT$V1)
  datT$X <- 2*(datT$X-min(datT$X))/(max(datT$X)-min(datT$X))-1
  names(datT)[names(datT) == "V1"]  <- var1
  names(datT)[names(datT) == "X"]  <- paste(var1,"X", sep="")
  datT
}

Features.PreProcessT <- function(datT){
  datT <- PreProcess.Continuous("Insured.age", datT)   
  datT <- PreProcess.Continuous("Car.age", datT)   
  datT <- PreProcess.Continuous("Credit.score", datT)   
  datT <- PreProcess.Continuous("Annual.miles.drive", datT)   
  datT <- PreProcess.Continuous("Years.noclaims", datT)   
  
  datT$MaritalX     <- as.integer(datT$Marital    ) - 0.5
  datT$Insured.sexX <- as.integer(datT$Insured.sex) - 0.5
  datT$RegionX      <- as.integer(datT$Region     ) - 0.5
  datT$CommuteX     <- model.matrix(~Car.use, data=datT)[,2] - 0.5
  datT$FarmerX      <- model.matrix(~Car.use, data=datT)[,3] - 0.5
  datT$PrivateX     <- model.matrix(~Car.use, data=datT)[,4] - 0.5
  datT$TerritoryX   <- as.integer(datT$Territory     ) - 1
  datT
}
datT <- Features.PreProcessT(datT)     

###############################################
#########  neural network (to find embeddings)
###############################################

# definition of feature variables (non-categorical)
features <- c(14:24)
(q0 <- length(features))

# learning data
Xlearn <- as.matrix(datT[, features])  # design matrix learning sample
Trlearn <- as.matrix(datT$TerritoryX)  
Ylearn <- as.matrix(datT$NB_Claim)

Vlearn <- as.matrix(log(datT$Duration))
lambda.hom <- sum(datT$NB_Claim)/sum(datT$Duration)

(TrLabel <- length(unique(datT$TerritoryX))) # 55

# define the network architecture
Design    <- layer_input(shape = c(q0),  dtype = 'float32', name = 'Design')
Territory <- layer_input(shape = c(1),   dtype = 'int32', name = 'Terriroty')
LogVol    <- layer_input(shape = c(1),   dtype = 'float32', name = 'LogVol')

TerritoryEmb1 = Territory %>% 
  layer_embedding(input_dim = TrLabel, output_dim = 1, input_length = 1, name = 'TerritoryEmb') %>%
  layer_flatten(name='Territory_flat')

Network1 = list(Design, TerritoryEmb1) %>% layer_concatenate(name='concate') %>% 
  layer_dense(units=10, activation='tanh', name='hidden1') %>%
  layer_dense(units=5, activation='tanh', name='hidden2') %>%
  layer_dense(units=3, activation='tanh', name='hidden3') %>%
  layer_dense(units=1, activation='linear', name='Network', 
              weights=list(array(0, dim=c(3,1)), array(log(lambda.hom), dim=c(1))))

Response1 = list(Network1, LogVol) %>% layer_add(name='Add') %>% 
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model_d1 <- keras_model(inputs = c(Design, Territory, LogVol), outputs = c(Response1))
model_d1 %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')
summary(model_d1)
# fitting the neural network
{t1 <- proc.time()
  fit_d1 <- model_d1 %>% fit(list(Xlearn, Trlearn, Vlearn), Ylearn, epochs=500, 
                       batch_size=1000, verbose=0, validation_split=0.2)
  (proc.time()-t1)}

plot(fit_d1)


TerritoryEmb2 = Territory %>% 
  layer_embedding(input_dim = TrLabel, output_dim = 2, input_length = 1, name = 'TerritoryEmb') %>%
  layer_flatten(name='Territory_flat')

Network2 = list(Design, TerritoryEmb2) %>% layer_concatenate(name='concate') %>% 
  layer_dense(units=10, activation='tanh', name='hidden1') %>%
  layer_dense(units=5, activation='tanh', name='hidden2') %>%
  layer_dense(units=3, activation='tanh', name='hidden3') %>%
  layer_dense(units=1, activation='linear', name='Network', 
              weights=list(array(0, dim=c(3,1)), array(log(lambda.hom), dim=c(1))))

Response2 = list(Network2, LogVol) %>% layer_add(name='Add') %>% 
  layer_dense(units=1, activation=k_exp, name = 'Response', trainable=FALSE,
              weights=list(array(1, dim=c(1,1)), array(0, dim=c(1))))

model_d2 <- keras_model(inputs = c(Design, Territory, LogVol), outputs = c(Response2))
model_d2 %>% compile(optimizer = optimizer_nadam(), loss = 'poisson')
summary(model_d2)
# fitting the neural network
{t2 <- proc.time()
  fit_d2 <- model_d2 %>% fit(list(Xlearn, Trlearn, Vlearn), Ylearn, epochs=500, 
                             batch_size=1000, verbose=0, validation_split=0.2)
  (proc.time()-t2)}

plot(fit_d2)

str(fit_d1)
str(fit_d2)


# extracting embedding weight for each territory
emb_Territory <- (model_d1$get_layer("TerritoryEmb") %>% get_weights())[[1]] %>%
  as.data.frame() 

datT$TerritoryEmb <- as.vector(model.matrix(~as.factor(TerritoryX)-1, data=datT) %*% as.matrix(emb_Territory))


###############################################
#########  GLM analysis
###############################################

datA <- cbind(datT[,c(1:11,26)], data[,12:52])

colnames(datA)[15:21]
colnames(datA)[c(22:24,27:28)]
colnames(datA)[30:41]
colnames(datA)[42:51]

Pct1learn  <- as.matrix(datA[, 15:21]) 
Pct1.pca <- prcomp(Pct1learn, scale = FALSE)
fviz_eig(Pct1.pca)
str(summary(Pct1.pca))
cumsum(summary(Pct1.pca)$importance[2,])

Pct2learn  <- as.matrix(datA[, c(22:24,27:28)]) 
Pct2.pca <- prcomp(Pct2learn, scale = FALSE)
fviz_eig(Pct2.pca)
str(summary(Pct2.pca))
cumsum(summary(Pct2.pca)$importance[2,])

Pct2.pca$rotation #Pct.drive.Xhours, Pct.drive.rushhours


Acbrlearn <- as.matrix(datA[, 30:41])  
Acbr.pca <- prcomp(Acbrlearn, scale = FALSE)
fviz_eig(Acbr.pca)
str(summary(Acbr.pca))
cumsum(summary(Acbr.pca)$importance[2,])

Acbr.pca$rotation #Accel.06miles, Brake.06miles, sums of Accel&Brake

datA$Acbr.others <- datA$Accel.08miles+datA$Accel.09miles+
  datA$Accel.11miles+datA$Accel.12miles+datA$Accel.14miles+
  datA$Brake.08miles+datA$Brake.09miles+datA$Brake.11miles+
  datA$Brake.12miles+datA$Brake.14miles


Turnlearn  <- as.matrix(datA[, 42:51])  
Turn.pca <- prcomp(Turnlearn, scale = FALSE)
fviz_eig(Turn.pca)
str(summary(Turn.pca))
cumsum(summary(Turn.pca)$importance[2,])

Turn.pca$rotation #Sum of left-turns, Sum of right-turns

datA$Left.turns <- datA$Left.turn.intensity08+datA$Left.turn.intensity09+
                   datA$Left.turn.intensity10+datA$Left.turn.intensity11+
                   datA$Left.turn.intensity12

datA$Right.turns <- datA$Right.turn.intensity08+datA$Right.turn.intensity09+
                    datA$Right.turn.intensity10+datA$Right.turn.intensity11+
                    datA$Right.turn.intensity12

PCtable <- rbind(
cumsum(summary(Pct1.pca)$importance[2,1:5]),
cumsum(summary(Pct2.pca)$importance[2,1:5]),
cumsum(summary(Acbr.pca)$importance[2,1:5]),
cumsum(summary(Turn.pca)$importance[2,1:5]))

rownames(PCtable) <- c("Pct1", "Pct2", "Acbr", "Turn")
PCtable


library(knitr)
library(kableExtra)
#options(knitr.table.format = "html")
options(knitr.table.format = "latex")

kable(PCtable, digits=4, escape=FALSE, booktabs = T,
      linesep = c("", "","", "","", "", "","", "\\hline", "","", "","", "","","", "","", "\\hline"))

rotable <- rbind(
Pct2.pca$rotation[,1:2],
Acbr.pca$rotation[,1:2],
Turn.pca$rotation[,1:2])

kable(rotable, digits=4, escape=FALSE, booktabs = T,
      linesep = c("", "","", "","", "", "","", "\\hline", "","", "","", "","","", "","", "\\hline"))

J=100
maxmat <- matrix(rep(NA, 5*J), ncol=5)
mxxmat <- matrix(rep(NA, 5*J), ncol=5)

system.time(
for (i in 1:J) {
print(i)
set.seed(i)
ll <- sample(c(1:nrow(datA)), round(0.9*nrow(datA)), replace = FALSE)
learn <- datA[ll,]
test <- datA[-ll,]

# Poisson GLM no territory embedding (territory df = 0)
d.glm1 <- glm(NB_Claim ~ Insured.age + Insured.sex + Car.age
              + Marital + Car.use + Credit.score + Region + Annual.miles.drive +
                Years.noclaims, 
              data=learn, offset=log(Duration), family=poisson())


# Poisson GLM without territory embedding (territory df = 54)
d.glm2 <- glm(NB_Claim ~ Insured.age + Insured.sex + Car.age
              + Marital + Car.use + Credit.score + Region + Annual.miles.drive +
                Years.noclaims + Territory, 
              data=learn, offset=log(Duration), family=poisson())

# Poisson GLM with territory embedding (territory df = 1)
  d.glm3 <- glm(NB_Claim ~ Insured.age + Insured.sex + Car.age
                + Marital + Car.use + Credit.score + Region + Annual.miles.drive +
                  Years.noclaims + TerritoryEmb, 
                data=learn, offset=log(Duration), family=poisson())

# Poisson GLM with telematics variable without dimension reduction
  d.glm4 <- glm(NB_Claim ~ Insured.age+Insured.sex+Car.age+Marital+Car.use+
                  Credit.score+Region+Annual.miles.drive+Years.noclaims+
                  TerritoryEmb+Annual.pct.driven+Total.miles.driven+
                  Pct.drive.mon+Pct.drive.tue+Pct.drive.wed+Pct.drive.thr+
                  Pct.drive.fri+Pct.drive.sat+Pct.drive.2hrs+Pct.drive.3hrs+
                  Pct.drive.4hrs+Pct.drive.rush.am+Pct.drive.rush.pm+
                  Avgdays.week+Accel.06miles+Accel.08miles+Accel.09miles+
                  Accel.11miles+Accel.12miles+Accel.14miles+Brake.06miles+
                  Brake.08miles+Brake.09miles+Brake.11miles+Brake.12miles+
                  Brake.14miles+Left.turn.intensity08+Left.turn.intensity09+
                  Left.turn.intensity10+Left.turn.intensity11+
                  Left.turn.intensity12+Right.turn.intensity08+
                  Right.turn.intensity09+Right.turn.intensity10+
                  Right.turn.intensity11+Right.turn.intensity12, 
                data=learn, offset=log(Duration), family=poisson())


# Poisson GLM with telematics variable with dimension reduction
  d.glm5 <- glm(NB_Claim ~ Insured.age+Insured.sex+Car.age+Marital+Car.use+
                  Credit.score+Region+Annual.miles.drive+Years.noclaims+
                  TerritoryEmb+Annual.pct.driven+Total.miles.driven+
                  Pct.drive.mon+Pct.drive.tue+Pct.drive.wed+Pct.drive.thr+
                  Pct.drive.fri+Pct.drive.sat+Pct.drive.rush.am+Pct.drive.rush.pm+
                  Avgdays.week+Accel.06miles+Brake.06miles+
                  Acbr.others+Left.turns+Right.turns, 
                data=learn, offset=log(Duration), family=poisson())

maxmat[i,] <- c(max(predict(d.glm1, newdata=test, type="response")),
  max(predict(d.glm2, newdata=test, type="response")),
  max(predict(d.glm3, newdata=test, type="response")),
  max(predict(d.glm4, newdata=test, type="response")),
  max(predict(d.glm5, newdata=test, type="response")))


mxxmat[i,] <- c(max(rpois(10000, predict(d.glm1, newdata=test, type="response"))),
                max(rpois(10000, predict(d.glm2, newdata=test, type="response"))),
                max(rpois(10000, predict(d.glm3, newdata=test, type="response"))),
                max(rpois(10000, predict(d.glm4, newdata=test, type="response"))),
                max(rpois(10000, predict(d.glm5, newdata=test, type="response"))))
})

library(scales)

ggplotColours <- function(n = 6, h = c(0, 360) + 15){
  if ((diff(h) %% 360) < 1) h[2] <- h[2] - 360/n
  hcl(h = (seq(h[1], h[2], length = n)), c = 100, l = 65)}

mycol <- ggplotColours(5)

plot( log(maxmat[,1]), ylim=c(-2,5), type='l', col=mycol[1],
      ylab="Max. log(lambda)", xlab=NULL)
lines(log(maxmat[,2]), ylim=c(-2,4), type='l', col=mycol[2])
lines(log(maxmat[,3]), ylim=c(-2,4), type='l', col=mycol[3])
lines(log(maxmat[,4]), ylim=c(-2,4), type='l', col=mycol[4])
lines(log(maxmat[,5]), ylim=c(-2,4), type='l', col=mycol[5])
abline(h=log(3), lty=2)
 legend("top", legend= c("Case 1", "Case 2", "Case 3", "Case 4", "Case 5"),
        col=mycol, lty=rep(1, 5), horiz=TRUE,
        cex=0.6, lwd=1, bty = "n")
 
emp_maxN <- rbind(
summary(mxxmat[,1]), 
summary(mxxmat[,2]),
summary(mxxmat[,3]), 
summary(mxxmat[,4]),
summary(mxxmat[,5])) 
rownames(emp_maxN) <- c("Case 1", "Case 2", "Case 3",
                        "Case 4", "Case 5")

kable(emp_maxN, digits=2, escape=FALSE, booktabs = T,
      linesep = c("", "","", "","", "", "","", "\\hline", "","", "","", "","","", "","", "\\hline"))

logLik(d.glm1)
logLik(d.glm2)
logLik(d.glm3)
logLik(d.glm4)
logLik(d.glm5)