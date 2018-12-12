#### Packages
library(dplyr)
library(randomForest)
library(MASS)
library(bigmemory)
library(data.table)
library(bigalgebra)
library(irlba)
library(glmnet)
library(lme4)
library(doParallel)
library(foreach)

#### Data input
load("train_data.RData")
load("test_data.RData")
train.x = train_data
test.x = test_data
train.y = read.csv("train_labels.csv", header = TRUE)
test.y = read.csv("test_labels.csv", header = TRUE)
train = cbind(volcanoe = train.y[,1], train_data)

#### Check dimension
#train image file (no header) : 7000*12100
#train label file (no header) : 7000*4
#test image file (no header) : 2734*12100
#test label file (header) : 2734*4
dim(train.x)
dim(test.x)
dim(train.y)
dim(test.y)

####check missing values
#only y has NAs
colnames(train.x)[colSums(is.na(train.x)) > 0]
colnames(train.y)[colSums(is.na(train.y)) > 0]
colnames(test.x)[colSums(is.na(test.x)) > 0]
colnames(test.y)[colSums(is.na(test.y)) > 0]

#### EDA
##plot
## rows that have volacoes: 1,10,16,30,35,39
train.x1 <- sapply(train.x[1:7000,], as.numeric)
volplot = function(a){
  m=matrix(a, nrow = 110, byrow = TRUE)
  image(m, col = grey((0:255)/255))
}
volplot(train.x1[1,])
volplot(train.x1[10,])
volplot(train.x1[16,])

####Preprocess
# Dimension reduction using SVD.
preprocess.svd = function(train, n.comp){
  train[is.na(train)] = 0
  z = svd(train[, 2:ncol(train)], nu=n.comp, nv=n.comp)
  s = diag(z$d[1:n.comp])
  train[, 2:ncol(train)] = z$u %*% s %*% t(z$v)
  train
}
n.comp = 12
preprocess.svd(train, n.comp)

#####Model

#Lasso
#current result: accuracy 92.7944 (lambda= 0.001882239)
train.x = as.matrix(train.x)
train.y=as.matrix(train.y[,1])
test.y = as.matrix(test.y[,1])
lasmodel = glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial")

pred = predict(lasmodel, newx=as.matrix(test.x), type = "response")
predfunc= function(a){
  ((sum(as.numeric(a > 0.5) == test.y[,1]))/ length(test.y[,1]))*100
}
pred.cal = apply(pred, 2, predfunc)
pred.cal[which.max(pred.cal)+1]
table(as.numeric(pred[,99] > 0.5), test.y[,1])

#Lasso - with tuning
#accuracy: 92.94% (lambda 0.001573354)
train.x = as.matrix(train.x)
train.y=as.matrix(train.y[,1])
cv.out = cv.glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial")
lam.seq = exp(seq(-6.5, -5, length=100))
lasmodel = glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial", lambda = lam.seq)
pred = predict(lasmodel, newx=as.matrix(test.x), type = "response")
predfunc= function(a){
  ((sum(as.numeric(a > 0.5) == test.y[,1]))/ length(test.y[,1]))*100
}
pred.cal = apply(pred, 2, predfunc)
pred.cal[which.max(pred.cal)]
table(as.numeric(pred[,97] > 0.5), test.y[,1])
#Lasso selection-linear regression
mylasso.coef.min = predict(lasmodel, s=cv.out$lambda.min, type="coefficients")
mylasso.coef.1se = predict(lasmodel, s=cv.out$lambda.1se, type="coefficients")
var.sel = row.names(mylasso.coef.1se)[nonzeroCoef(mylasso.coef.1se)[-1]]
tmp.X = X[, colnames(X) %in% var.sel]


#logistic (marginal screening)
cl <- makeCluster(3)
registerDoParallel(cl)

pre = Sys.time()
vol_results <- foreach(j = 1:ncol(train.x), .packages = 'lme4', .combine='rbind') %dopar% {
  
  pix = scale(train.x[,j])
  fit.glm <- summary(glm(as.factor(train.y) ~ pix, family = "binomial"))
  fit.glm$coefficients[2,4]
}

Sys.time() - pre

stopCluster(cl)

rownames(vol_results) = colnames(train.x)
colnames(vol_results) = "glm.p value"

#save(vol_results, file = "vol_result.RData")
#load("vol_result.RData")

totalpix = 400
sortedresults = as.matrix(vol_results[order(vol_results),])
usepix = names(sortedresults[1:totalpix,])

top_loc = match(usepix, rownames(vol_results))

pic_m_c = matrix(0,110,110)
pic_m_c[top_loc] = 1
image(pic_m_c,col = grey(12100:0/12100),main = "Marginal Screening index for classficiation")

sel.pix = train.x[,top_loc]

las.pix = train.x[,las]


training = data.frame(train.y,sel.pix)
colnames(training) = c("volcano",colnames(sel.pix))

test.x = as.matrix(test.x)

test.pip_glm = test.x[,top_loc]
test.pipx = test.x[,top_loc]



logmodel = glm(as.factor(volcano)~., data = training, family = "binomial")

train.error = (predict(logmodel, data.frame(sel.pix), type = "response") > 0.5) #0.9148571
log_pred = (predict(logmodel, data.frame(test.pipx), type = "response") > 0.5)

table(log_pred,test.y) # accuracy = 0.892831




load("lassomodel.RData")

las_cla_index = which(lasmodel$beta!=0)
pic_las_c = matrix(0,110,110)
pic_las_c[las_cla_index] = 1
image(pic_las_c,col = grey(12100:0/12100),main = "Lasso index for classification")


#LDA
ldmodel = lda(train.x, train.y)

#random forest
rfmodel = randomForest(train.x, y=train.y[,1], ntree = 200)

#neural network

#xgboost
#accuracy: 93.56255
library(xgboost)
train.x = as.matrix(train.x)
train.y=as.matrix(train.y[,1])
params <- list(
  eta = 0.1,
  max_depth = 7,
  min_child_weight =5,
  subsample = 0.65,
  colsample_bytree = 0.8,
  silent = 1
)
xgb.fit.final <- xgboost(
  params = params,
  data = as.matrix(train.x),
  label = train.y,
  nrounds = 200,
  objective = 'binary:logistic',
  verbose = 0
)

xgpred = predict(xgb.fit.final, as.matrix(test.x))

predfunc= function(a){
  mean(as.numeric(a > 0.5) == test.y[,1])*100
}
xgpred.cal = mean(as.numeric(xgpred > 0.5) == test.y[,1])*100
xgpred.cal


##xgtuning
#tune
hyper_grid <- expand.grid(
  eta = c(.05, .1, .3),
  max_depth = c(3, 5, 7),
  min_child_weight = c(3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# grid search 
start.time = proc.time()

for(i in 1:nrow(hyper_grid)) {
  print(i)
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  # reproducibility
  set.seed(7607)
  
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = as.matrix(train.x),
    label = train.y,
    nrounds = 1000,
    nfold = 5,
    objective = 'binary:logistic',
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_error_mean)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_error_mean)
}

total.time= proc.time() - start.time


hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)

