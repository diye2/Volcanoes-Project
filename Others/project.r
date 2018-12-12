library(data.table)
library(MASS)
library(ggplot2)
library(glmnet)
library(doParallel)
library(foreach)


train_y <- read.csv("train_labels.csv", header = TRUE)
test_y <- read.csv("test_labels.csv", header = TRUE)
load("train_data.RData")
load("test_data.RData")

# for classification
train.x = as.matrix(train_data)
train.y=as.matrix(train_y[,1])
test.x = as.matrix(test_data)
test.y = as.matrix(test.y[,1])

## Lasso model selection:
cv.out = cv.glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial")
lam.seq = exp(seq(-6.5, -5, length=100))
lasmodel = glmnet(x=train.x, y=train.y, alpha = 1, family = "binomial", lambda = lam.seq)
las_cla_index = which(lasmodel$beta!=0)

mylasso.coef.min = predict(lasmodel, s=cv.out$lambda.min, type="coefficients")
mylasso.coef.1se = predict(lasmodel, s=cv.out$lambda.1se, type="coefficients")
var.sel = row.names(mylasso.coef.1se)[nonzeroCoef(mylasso.coef.1se)[-1]]
tmp.X = X[, colnames(X) %in% var.sel]

### Lasso prediction:

pred = predict(lasmodel, newx=as.matrix(test.x), type = "response")
predfunc= function(a){
  ((sum(as.numeric(a > 0.5) == test.y[,1]))/ length(test.y[,1]))*100
}
pred.cal = apply(pred, 2, predfunc)
pred.cal[which.max(pred.cal)]
table(as.numeric(pred[,97] > 0.5), test.y[,1])

## Marginal Screening variable selection:

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
#save(vol_results, file = "vol_result.RData")

rownames(vol_results) = colnames(train.x)
colnames(vol_results) = "glm.p value"

totalpix = 400
sortedresults = as.matrix(vol_results[order(vol_results),])
usepix = names(sortedresults[1:totalpix,])

top_pix = match(usepix, rownames(vol_results))

### Logistic regression prediction using Lasso:
las.pix = train.x[,las_cla_index]
training_L = data.frame(train.y,las.pix)
colnames(training_L) = c("volcano",colnames(las.pix))
test.lasx = test.x[,las.pix]
las_pred = (predict(lasmodel, data.frame(test.lasx), type = "response") > 0.5)
table(las_pred,test.y)

### Logistic regression prediction using Marginal Screeing:
sel.pix = train.x[,top_pix]
training_M = data.frame(train.y,sel.pix)
colnames(training_M) = c("volcano",colnames(sel.pix))
test.pipx = test.x[,top_loc]

logmodel = glm(as.factor(volcano)~., data = training, family = "binomial")
train.error = (predict(logmodel, data.frame(sel.pix), type = "response") > 0.5)
log_pred = (predict(logmodel, data.frame(test.pipx), type = "response") > 0.5)
table(log_pred,test.y)


# for Poisson prediction
train_num = as.matrix(train_y[,4])
train_num[train_num == "NaN"] = 0
test_num = as.matrix(test_y[,4])
test_num[test_num == "NaN"] = 0

## Lasso model selection:

mycv = cv.glmnet(x= train.x, y = train_num, family = "poisson",alpha = 1)
lam.seq = exp(seq(-5, -4.5, length=100))
cv.out = cv.glmnet(x=train.x, y=train_num, alpha = 1, family = "poisson", lambda = lam.seq)
las_num = glmnet(x=train.x, y=train_num, alpha = 1, family = "poisson", lambda = cv.out$lambda.min)
#save(las_num, file = "las_num.RData")
las_index = which(las_num$beta != 0)

### Lasso Prediction:

pred = predict(las_num, newx=as.matrix(test_data), type = "response") #RMSE = 0.4216523
sqrt(mean((pred - test_num)^2))


## Marginal Screening variable selection:
cl <- makeCluster(3)
registerDoParallel(cl)

pre = Sys.time()
glm_num <- foreach(j = 1:ncol(train.x), .packages = 'lme4', .combine='rbind') %dopar% {
  
  pix = scale(train.x[,j])
  fit.glm <- summary(glm(train_num ~ pix, family = "poisson"))
  fit.glm$coefficients[2,4]
}

Sys.time() - pre

stopCluster(cl)

rownames(glm_num) = colnames(train.x)
colnames(glm_num) = "glm.p value"

#save(glm_num, file = "glm_nunm_result.RData")

totalpix = 250
sortedresults = as.matrix(glm_num[order(glm_num),])
usepix = names(sortedresults[1:totalpix,])

top_pix_num = match(usepix, rownames(glm_num))

### Logistic regression prediction using Lasso:

las_pix= train.x[,las_index]
training_las = data.frame(train_num,las_pix)
colnames(training_las) = c("volcano_num",colnames(las_pix))
test.las = test.x[,las_index]

glm.las.num = glm(volcano_num~., data = training_las, family = "poisson")
glm.las_pred = predict(glm.las.num, data.frame(test.las), type = "response") 
sqrt(mean((glm.las_pred - test_num)^2)) #RMSE = 0.6798163

### Logistic regression prediction using Marginal Screening:
sel.pix = train.x[,top_pix_num]
training = data.frame(train_num,sel.pix)
colnames(training) = c("volcano_num",colnames(sel.pix))
test.pipx = test.x[,top_pix_num]

glm.num = glm(volcano_num~., data = training, family = "poisson")
log_pred = predict(glm.num, data.frame(test.pipx), type = "response")
sqrt(mean((log_pred - test_num)^2)) # 0.5072927

glm_result = list(glm.las.num, glm.num)
#save(glm_result, file = "glm_num.RData")




pic_M = matrix(0,110,110)
pic_M[top_pix_num] = 1
image(pic_M,col = grey(12100:0/12100),main = "Marginal Screening index for Poisson prediction")


pic_las = matrix(0,110,110)
pic_las[las_index] = 1
image(pic_las,col = grey(12100:0/12100),main = "lasso index for Poisson prediction")



myimport = rfmodel$importance
my_matrix = matrix(myimport[,4],110,110,byrow = T)

library(RColorBrewer)
pal <- colorRampPalette(brewer.pal(11, "RdYlGn"))(100)
mycol <- c("black", "yellow") 
pal <- colorRampPalette(mycol)(100)
heatmap(my_matrix,Rowv = NA, Colv = NA,revC = T,col = pal)


