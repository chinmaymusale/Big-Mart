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
ohe = dummyVars("~.", data = dataset[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")], fullRank = T) 
ohe_df = data.table(predict(ohe, dataset[,-c("Item_Identifier", "Outlet_Establishment_Year", "Item_Type")])) 
dataset = cbind(dataset[,"Item_Identifier"], ohe_df)


