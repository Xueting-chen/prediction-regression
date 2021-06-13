
df1 <- read.csv('student-mat.csv',sep=';', stringsAsFactors = T)

library(ggplot2)
library(corrplot)
library(caret)
library(dummies)
library(caTools)
library(glmnet)
library(randomForest)
library(leaps)

head(df1)
#To remove variables G2 and G1
df1 = subset(df1, select=-c(G1,G2))
head(df1)
# Check for NA values
any(is.na(df1))
str(df1)

##Exploratory Data Analysis
#Selecting the numeric columns only
num.cols <- sapply(df1, is.numeric)

#Filtering out numeric columns for correlation
cor.data <- cor(df1[,num.cols])

#Correlation plot
corrplot::corrplot(cor.data, method='color')

#Analysis of G3 score
G3_score.math <- df1$G3
hist(G3_score.math)
qplot(G3, data=df1, geom='histogram', bins=20, fill=..count.., xlab='G3 (final grade)')

# glmnet() require matrix ==> Need to create dummies for categorical X, model.matrix will change automatically                                 
math <- model.matrix(~., data = df1)[,-1]
head(math[,-1])
math.df <- data.frame(math)

# Set the seed
set.seed(100)

# Split the data into test and train sets
sample <- sample.split(Y = df1$G3, SplitRatio = 0.7)

# Training data
train_m <- subset(math.df, sample == T)

# Test data
test_m  <- subset(math.df, sample == F)

model_m <- lm(G3~., train_m)
alias(model_m)
summary(model_m)

# Grab residuals
res_m <- residuals(model_m)

# Convert to data frame for plot
res_m <- as.data.frame(res_m)

# Plot residuals
qplot(res_m, data=res_m, geom='histogram')

par(mfrow=c(2,2))
plot(model_m)

G3.predictions.m <- predict(model_m, newdata = test_m)
summary(G3.predictions.m)

results.m <- cbind(G3.predictions.m, test_m$G3)
colnames(results.m) <- c('Predicted', 'Actual')
results.m <- as.data.frame(results.m)
# LM accuracy:
data.frame(
  RMSE1= RMSE(G3.predictions.m, test_m$G3),
  Rsquare1 = R2(G3.predictions.m, test_m$G3))

# R2 = 0.1540458 very low prediction accuracy
# RMSE = 4.222536

#--------------------------------------------------------------------

#Backward Selection
set.seed(100)
# Set up repeated k-fold cross-validation
train.control <- trainControl(method = "cv", number = 10)
# Train the model
step.model1 <- train(G3 ~., data = df1,
                     method = "leapBackward", 
                     tuneGrid = data.frame(nvmax = 1:30),
                     trControl = train.control)
# Result of the backward stepwise regression
step.model1$results
# Best model
step.model1$bestTune #RMSE = 4.26
summary(step.model1$finalModel)
# Coefficients of best model
coef(step.model1$finalModel, 4) #4 predictors selected

#--------------------------------------------------------------------
#Ridge regression 

set.seed(100)
x.m =model.matrix(G3~.,data=train_m)[,-1]
y.m =train_m$G3
grid = 10^seq(10,-2,length=100)

lm_ridge <- glmnet(x.m,y.m,alpha=0, lambda = grid)
summary(lm_ridge)
dim(coef(lm_ridge))

cv_fit = cv.glmnet(x.m,y.m,alpha=0)
plot(cv_fit)
opt_lambda = cv_fit$lambda.min #4.35
model.math <- glmnet(x.m, y.m, alpha = 0, lambda = cv_fit$lambda.min)
coef(model.math)

x.test <- model.matrix(G3 ~., test_m)[,-1]
predictions <- model.math %>% predict(x.test) %>% as.vector()
data.frame(
  RMSE = RMSE(predictions, test_m$G3), # RMSE = 4.167136
  Rsquare = R2(predictions, test_m$G3)) #0.1576634

#-----------------------------------------------------------
# Random Forest Regression
set.seed(100)
math.rf <- randomForest(G3~., data=train_m, ntree=500, mtry=10) 
math.model <-predict(math.rf, test_m)
# RF model accuracy
data.frame(
  RMSE = RMSE(math.model , test_m$G3),  #3.688
  Rsquare = R2(math.model , test_m$G3)  #0.384
)

#Cross-check results
math.results <- cbind(math.model, test_m$G3)
colnames(math.results) <- c('Predicted', 'Actual')
math.results <- as.data.frame(math.results)
rss3 <- sum((math.results$Predicted - math.results$Actual)^2)
tss3 <- sum((mean(df1$G3) - math.results$Actual)^2)
R2.math.rf <- 1-rss3/tss3
# R-squared for RF = 33.5%
RMSE.rf.Math <- sqrt(mean((math.model - test_m$G3)^2))
#RMSE = 3.688

## Random Forest has the lowest RMSE


