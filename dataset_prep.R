library(dplyr)

setwd("/home/admin123/Overview_BDML/example")
fp = "housing.data"
df = read.table(fp)
col.names = c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS","RAD", "TAX",
              "PTRATIO", "B", "LSTAT", "MEDV")
names(df) = col.names
df.train<-sample_frac(df, 0.667)
train.ind<-as.numeric(rownames(df.train)) # because rownames() returns character
df.test<-df[-train.ind,]
fp.train = "boston_housing_train.csv"
fp.test = "boston_housing_test.csv"
write.csv(df.train, fp.train, row.names = FALSE)
write.csv(df.test, fp.test, row.names = FALSE)