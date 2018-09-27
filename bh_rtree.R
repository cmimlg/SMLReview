library(rpart)
library(Metrics)
library(ggplot2)
library(xtable)
setwd("/home/admin123/Overview_BDML/example")
fp.train = "boston_housing_train.csv"
fp.test = "boston_housing_test.csv"
df.train = read.csv(fp.train)
df.test = read.csv(fp.test)
# grow tree 
fit <- rpart(MEDV~., 
             method="anova", data=df.train)
preds = setdiff(names(df.test), "MEDV")
df.preds = df.test[preds]
ytp = predict(fit, df.preds)
yact = df.test["MEDV"]
err.rmse = rmse(yact, ytp)
tree.cp.df = as.data.frame(fit$cptable)
req.cols = c("CP", "nsplit", "xerror")
tree.cp.df = tree.cp.df[req.cols]
print(xtable(tree.cp.df), include.rownames = FALSE)
printcp(fit) # display the results 
xtable(fit$cptable)
grid(col = "black")

summary(fit) # detailed summary of splits

# prune the tree 
pfit<- prune(fit, cp=0.032) # from cptable   

# plot the pruned tree 
plot(pfit, uniform=TRUE, 
     main="Pruned Regression Tree for Boston Housing")

text(pfit, use.n=TRUE, all=TRUE, cex=.8)
ytp = predict(pfit, df.preds)
df.pred = as.data.frame(cbind(ytp, yact))
names(df.pred) = c("Pred", "Test")
fp.out = "bh_test_pred.csv"
write.csv(df.pred, fp.out, row.names = FALSE)