install.packages("tidyverse")
install.packages("mlbench")
install.packages("Hmisc")
install.packages("data.table",
                 repos="http://R-Forge.R-project.org")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("corrplot")
install.packages("xgboost")
install.packages("cowplot")
library(data.table)
library(dplyr)  
library(ggplot2)  
library(caret)    
library(corrplot) 
library(xgboost)    
library(cowplot) 
library(plyr)
library(Hmisc)


#importing dataset

train <- read.csv(file.choose())
test <- read.csv(file.choose())
submission <- read.csv(file.choose())

#set working directory
setwd("C:\\Users\\DELL\\Desktop\\big_mart")
getwd()

#exploring the data
nrow(train)
nrow(test)
ncol(train)
ncol(test)
head(train)
head(test)
str(train)
str(test)
summary(train)
summary(test)

#combining train and test dataset
dataset <- rbind.fill(train, test)
str(dataset)
summary(dataset)
factor(dataset$Outlet_Establishment_Year)
dataset$Outlet_Establishment_Year <- factor(dataset$Outlet_Establishment_Year)

#Exploratory Data Analysis- Univariate analysis
ggplot(train, aes(x=Item_Outlet_Sales)) + geom_histogram() #target variable is right skewed

#Independent variable(numeric)
ggplot(dataset, aes(x=Item_Weight)) + geom_histogram(binwidth = 0.5)
ggplot(dataset, aes(x=Item_Visibility)) + geom_histogram(binwidth = 0.005)
ggplot(dataset, aes(x=Item_MRP)) + geom_histogram(binwidth = 1)

#Independent variable(categorical)
dataset$Item_Fat_Content[dataset$Item_Fat_Content=="LF"] = "Low Fat"
dataset$Item_Fat_Content[dataset$Item_Fat_Content=="low fat"] = "Low Fat"
dataset$Item_Fat_Content[dataset$Item_Fat_Content=="reg"] = "Regular"
ggplot(dataset, aes(x=Item_Fat_Content)) + geom_bar()
ggplot(dataset, aes(x=Item_Type)) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(x=Outlet_Establishment_Year)) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(x=Outlet_Size)) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(dataset$Outlet_Location_Type)) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(dataset$Outlet_Type)) + geom_bar() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Exploratory Data Analysis- Bivariate analysis - Target Variable vs Independent Variable
ggplot(dataset, aes(Item_Weight, Item_Outlet_Sales)) + geom_point() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Item_Visibility, Item_Outlet_Sales)) + geom_point() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Item_MRP, Item_Outlet_Sales)) + geom_point() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Item_Type, Item_Outlet_Sales)) + geom_violin() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Item_Fat_Content, Item_Outlet_Sales)) + geom_violin() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Outlet_Identifier, Item_Outlet_Sales)) + geom_violin() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Outlet_Establishment_Year, Item_Outlet_Sales)) + geom_violin() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Item_Weight, Item_Outlet_Sales)) + geom_point() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Outlet_Location_Type, Item_Outlet_Sales)) + geom_violin() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(dataset, aes(Outlet_Type, Item_Outlet_Sales)) + geom_violin() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Missing value treatment
sum(is.na(dataset$Item_Weight))
sum(is.na(dataset$Item_Visibility))
dataset$Item_Weight <- impute(dataset$Item_Weight, mean)
zero_index = which(dataset$Item_Visibility == 0)
for (i in zero_index) {
  item = dataset$Item_Visibility[i]
  dataset$Item_Visibility[i] = mean(dataset$Item_Visibility[dataset$Item_Identifier == item], na.rm = T)
}

#Feature Engineering
perishable = c("Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood")
non_perishable = c("Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household", "Soft Drinks")

dataset$Item_Type_new <- ifelse(dataset$Item_Type %in% perishable, "perishable",
                              ifelse(dataset$Item_Type %in% non_perishable, "non_perishable",
                                     "not_sure"))
table(dataset$Item_Type, substr(dataset$Item_Identifier, 1, 2))
dataset$Item_Category <- substr(dataset$Item_Identifier, 1, 2)

dataset$Item_Fat_Content[dataset$Item_category == "NC"] = "Non-Edible" 
dataset$Outlet_Years <- 2013 - dataset$Outlet_Establishment_Year 
dataset$Outlet_Establishment_Year = as.factor(dataset$Outlet_Establishment_Year) 
dataset$price_per_unit_wt <- (dataset$Item_MRP)/(dataset$Item_Weight)

#Encoding Categorical variable
dataset$Outlet_Size_num <- ifelse(dataset$Outlet_Size == "Small", 0, ifelse(dataset$Outlet_Size == "Medium", 1, 2))
dataset$Outlet_Location_Type_num <- ifelse(dataset$Outlet_Location_Type == "Tier 3", 0, ifelse(dataset$Outlet_Location_Type == "Tier 2", 1, 2))
dataset$Outlet_Size <- NULL
dataset$Outlet_Location_Type <- NULL

#One Hot Encoding
setDT(dataset)
ohe = dummyVars("~.", data = dataset[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T) 
ohe_df = data.table(predict(ohe, dataset[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")])) 
dataset = cbind(dataset[,"Item_Identifier"], ohe_df)

#Preprocessing data
dataset$Item_Visibility <- log(dataset$Item_Visibility + 1)
dataset$price_per_unit_wt <- log(dataset$price_per_unit_wt + 1)

num_vars = which(sapply(dataset, is.numeric))
num_vars_names = names(num_vars)
dataset_numeric = dataset[, setdiff(num_vars_names, dataset$Item_Outlet_Sales), with = F]
prep_num = preProcess(dataset_numeric, method = c("center", "scale"))
dataset_numeric_norm = predict(prep_num, dataset_numeric)
dataset[, setdiff(num_vars_names, dataset$Item_Outlet_Sales) := NULL]
dataset = cbind(dataset, dataset_numeric_norm)

#Splitting the combined dataset

train = dataset[1:nrow(train)]
test = dataset[(nrow(train) + 1):nrow(dataset)]
test[,Item_Outlet_Sales := NULL]

#Correlated Variables 
cor_train = cor(train[, -c("Item_Identifier")])
corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)

#Model Building - Linear Regression
linear_reg_mod = lm(Item_Outlet_Sales ~ ., data = train[, -c("Item_Identifier")])
submission$Item_Outlet_Sales = predict(linear_reg_mod, test[, -c("Item_Identifier")])
write.csv(submission, "Linear_Reg_submit.csv", row.names = F)

#Model Building - Regularized Linear Regression
set.seed(1235)
my_control = trainControl(method = "cv", number = 5)
Grid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, by = 0.0002))
lasso_linear_reg_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")], y = train$Item_Outlet_Sales, method='glmnet', trControl=my_control, tuneGrid=Grid)


#Model Building - Random Forrest
set.seed(1237)
my_control = trainControl(method = "cv", number=5)
tgrid = expand.grid(
  .mtry = c(3:10),
  .splitrule = "variance", 
  .min.node.size = c(10, 15, 20)
)
rf_mod = train(x = train[, -c("Item_Identifier", "Item_Outlet_Sales")],
               y = train$Item_Outlet_Sales, 
               method = 'ranger',
               trControl = my_control,
               tuneGrid = tgrid,
               num.trees = 400,
               importance = "permutation"
               )

#Model Building - XGBoost
param_list = list(objective = "reg:linear", eta=0.01, gamma = 1, max_depth=6, subsample=0.8, colsample_bytree=0.5)
dtrain = xgb.DMatrix(data = as.matrix(train[,-c("Item_Identifier", "Item_Outlet_Sales")]), label= train$Item_Outlet_Sales)
dtest = xgb.DMatrix(data = as.matrix(test[,-c("Item_Identifier")]))               

set.seed(112)
xgbcv = xgb.cv(params = param_list, data = dtrain, nrounds = 1000, nfold = 5, print_every_n = 10, early_stopping_rounds = 30, maximize = F)
xgb_model = xgb.train(data = dtrain, params = param_list, nrounds = 430)
var_imp = xgb.importance(feature_names = setdiff(names(train), c("Item_Identifier", "Item_Outlet_Sales")), model = xgb_model) 
xgb.plot.importance(var_imp)

