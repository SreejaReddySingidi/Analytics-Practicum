'Name: Sreeja Reddy Singidi
Last updated: 05/08/2024'


'Note: Approximate time taking to run the entire code = 5-6 minutes'

#Libraries required
library(psych)
library(gridExtra)
library(ggplot2)
library(caret)
library(RColorBrewer)
library(MASS)
library(rpart)
library(rpart.plot)
library(class)
library(gains)
library(dplyr)
library(reshape2)
library(glmnet)
library(Metrics)
library(Boruta)
library(randomForest)
library(rattle)
library(neuralnet)
library(mice)
library(corrplot)
library(e1071)
library(caret)
library(sqldf)

'#########################################################################################################################################
__________________________________________________________________________________________________________________________________

                  PROJECT  1: MAXIMIZING PROFITS FROM SOFTWARE SALES
___________________________________________________________________________________________________________________________________
##########################################################################################################################################'

#****************************COLLECTING DATA****************************#
north_point_data <- read.csv("North-Point List.csv")
str(north_point_data)
dim(north_point_data)

#****************************CHECKING MISSING VALUES****************************#
sum(is.na(north_point_data))

#****************************CHECKING 0 VALUES****************************#
#Selecting only the numerical attributes
numerical_attributes <- north_point_data[, c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")]
zero_values <- colSums(numerical_attributes == 0)
zero_values
#--------------Count of purchasers and non-purchasers-------------#
purchase_counts <- table(north_point_data$Purchase)
#Check if purchase=0, then spending =0 or not
zero_count <- sum(north_point_data$Purchase == 0 & north_point_data$Spending == 0)
#Check if purchase=1, spending=0
one_count <- sum(north_point_data$Purchase == 1 & north_point_data$Spending == 0)
#Check if Purchase=0 but has spending
unwanted_spending <- sum(north_point_data$Purchase == 0 & north_point_data$Spending > 0)
nonpurchase_spending <- north_point_data[north_point_data$Purchase == 0 & north_point_data$Spending != 0, ]

#****************************Attribute Analysis****************************#
#--------------DISTRIBUTION OF OUTCOME VARIABLES-------------#
#Purchase
colors <- c("chartreuse", "blue4")
barplot(table(north_point_data$Purchase), main = "Distribution of Purchase", xlab = "Purchase", ylab = "Frequency", col = colors)
#Spending
hist(north_point_data$Spending, main= "Distribution of Spending", xlab = "Freq", col = 'purple')

#--------------DISTRIBUTION OF CATEGORICAL VARIABLES-------------#
#US
ggplot(north_point_data, aes(x = US)) +
  geom_bar(fill = "darkolivegreen2") +
  geom_text(stat = 'count', aes(label = ..count..)) +  
  labs(title = "Distribution of attribute 'US' ")

#Web Order
ggplot(north_point_data, aes(x = Web.order)) +
  geom_bar(fill = "darkolivegreen2") +
  geom_text(stat = 'count', aes(label = ..count..)) +  
  labs(title = "Distribution of Web Order")

#Gender
ggplot(north_point_data, aes(x = Gender.male)) +
  geom_bar(fill = "darkolivegreen2") +
  geom_text(stat = 'count', aes(label = ..count..)) +  
  labs(title = "Distribution of Gender")

#Address
ggplot(north_point_data, aes(x = Address_is_res)) +
  geom_bar(fill = "darkolivegreen2") +
  geom_text(stat = 'count', aes(label = ..count..)) +  
  labs(title = "Distribution of Address")

#--------------DISTRIBUTION OF NUMERIC ATTRIBUTES-------------#
par(mfrow = c(2, 2))  

# Create the histogram
hist(north_point_data$Freq, main = "Frequency", col = "yellow2", border = "black")
hist(north_point_data$last_update_days_ago, main = "Last Update", col = "yellow2", border = "black")
hist(north_point_data$X1st_update_days_ago, main = "First Update", col = "yellow2", border = "black")
hist(north_point_data$Spending, main = "Spending", col = "yellow2", border = "black")

par(mfrow = c(1, 1))

#--------------DISTRIBUTION OF ALL SOURCES WHEN PURCHASED-------------#
source_attributes <- c("source_a", "source_c", "source_b", "source_d", "source_e", "source_m","source_o", "source_h", "source_r", "source_s", "source_t", "source_u", "source_p", "source_x", "source_w")
#Calculate the count of '1' for each source
source_counts <- sapply(source_attributes, function(source) sum(north_point_data$Purchase == 1 & north_point_data[[source]] == 1))
bar_colors <- rainbow(length(source_counts))
plot_source_counts <- barplot(source_counts, col = bar_colors, main = "Counts of Purchase (Purchase = 1) for Each Source", ylab = "Count")

#--------------ANALYZING THE VARIABLE FREQUENCY WITH OUTCOME VARIABLES-------------#
#Purchase vs Frequency bar plot
ggplot(north_point_data, aes(x = as.factor(Purchase), fill = as.factor(Freq))) +
  geom_bar(position = "dodge") +
  labs(title = "Bar Plot of Purchase vs Frequency",
       x = "Purchase",
       y = "Frequency",
       fill = "Frequency")

#Relationship between Freq and Spending
ggplot(north_point_data, aes(x = Freq, y = Spending)) +
  geom_point() +
  labs(title = "Scatter Plot of Frequency vs Spending",
       x = "Frequency",
       y = "Spending")

#--------------AVERAGE SPENDING BY GENDER-------------#
barplot(height = tapply(north_point_data$Spending, north_point_data$Gender.male, mean),
        names.arg = c("Other Gender", "Male"),
        col = c("chartreuse", "blue4"),
        main = "Average Spending by Gender",
        xlab = "Gender",
        ylab = "Average Spending")

#--------------BOXPLOTS FOR NUMERIC ATTRIBUTE-------------#
numeric_attributes <- c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")
boxplot(north_point_data[, numeric_attributes], main = "Box plots for Numeric Attributes")

#--------------SCATTER PLOT MATRIX-------------#
pairs.panels(north_point_data[c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")])

#****************************PREDICTOR ANALYSIS AND RELEVANCY****************************#
#--------------Removing sequence number-------------#
north_point_data <- subset(north_point_data, select = -c(sequence_number)) #sequence number is not relevant to build the model, so eliminate

#--------------Identify rows with all source attributes = 0-------------#
source_cols <- names(north_point_data)[startsWith(names(north_point_data), "source_")]
zero_sources_rows <- north_point_data[rowSums(north_point_data[source_cols]) == 0, ]
zero_sources_rows
nrow(zero_sources_rows)
#New dataframe
no_source_data <- north_point_data[rowSums(north_point_data[source_cols]) == 0, ]
no_source_data$source_category <- "No_Source_Data"
nrow(no_source_data)

#****************************DATA TRANSFORMATIONS****************************#
names(north_point_data)[names(north_point_data) == "Freq"] <- "Frequency"
names(north_point_data)[names(north_point_data) == "last_update_days_ago"] <- "Days_since_Last_Update"
names(north_point_data)[names(north_point_data) == "X1st_update_days_ago"] <- "Days_since_First_Update"
names(north_point_data)[names(north_point_data) == "Web.order"] <- "online_order"
names(north_point_data)[names(north_point_data) == "Gender.male"] <- "gender_male"

#****************************DIMENSION REDUCTION****************************#
#--------------Correlation for numeric attributes-------------#
numerical_attributes_matrix <- north_point_data[, c("Frequency", "Days_since_Last_Update", "Days_since_First_Update")]
cor(numerical_attributes_matrix)

#****************************Maximum spending amount****************************#
max(north_point_data$Spending)

#****************************DATA PARTITIONING****************************#
set.seed(2024)
indices <- sample(1:2000)
#Training (800 records)
train_data <- north_point_data[indices[1:800], ]
#Validation (700 records)
validation_data <- north_point_data[indices[801:1500], ]
#Test (500 records)
test_data <- north_point_data[indices[1501:2000], ]
dim(north_point_data)
dim(train_data)
dim(validation_data)
dim(test_data)

#****************************GOAL 1- CLASSIFICATION****************************#
#--------------Logistic Regression-------------#
initial_model <- glm(Purchase ~ . - Spending, data = train_data, family = binomial)
summary(initial_model)
valid_predict_initial <- predict(initial_model, newdata = validation_data, type = "response")

#Confusion matrix for the initial model
valid_predict_initial <- factor(ifelse(valid_predict_initial > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
valid_predict_initial <- factor(valid_predict_initial, levels = c("1", "0"))
validation_data$Purchase <- factor(validation_data$Purchase, levels = c("1", "0"))

cm_initial <- confusionMatrix(valid_predict_initial, validation_data$Purchase, positive="1")
cm_initial

#--------------Step-wise-------------#
stepwise_model<- stepAIC(initial_model, direction = "backward")

#--------------Model 1-------------#
model_1 <- glm(Purchase ~ source_a + source_e + source_h + source_r + source_s + 
                 source_t + source_u + source_p + source_x + source_w + Frequency + 
                 Days_since_Last_Update + online_order + Address_is_res, data = train_data, family = binomial)
summary(model_1)
validation_data$Purchase <- factor(validation_data$Purchase, levels = levels(factor(train_data$Purchase)))

selected_variables_1 <- c("source_a", "source_e", "source_h", "source_r", "source_s",  
                          "source_t", "source_u", "source_p", "source_x", "source_w", "Frequency", 
                          "Days_since_Last_Update", "online_order", "Address_is_res", "Purchase")
selected_validation_data <- validation_data[, selected_variables_1]
model1_predict <- predict(model_1, newdata = selected_validation_data, type = "response")
model1_predict <- factor(ifelse(model1_predict > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
model1_predict <- factor(model1_predict, levels = c("1", "0"))
selected_validation_data$Purchase <- factor(selected_validation_data$Purchase, levels = c("1", "0"))
cm_1 <- confusionMatrix(model1_predict, selected_validation_data$Purchase)
cm_1

#--------------Model 2-------------#
#Significant variable set 1
selected_variables_2 <- c("source_a", "source_e", "source_h", "source_r","source_t", "source_u", "source_p", "source_w", 
                          "Frequency", "Days_since_Last_Update", "online_order", "Address_is_res", "Purchase")

model_2 <- glm(Purchase ~ source_a + source_e + source_h + source_r + source_t + 
                 source_u + source_p +source_w + Frequency + 
                 Days_since_Last_Update + online_order + Address_is_res, data = train_data, family = binomial)

summary(model_2)

selected_validation_data2 <- validation_data[, selected_variables_2]
model2_predict <- predict(model_2, newdata = selected_validation_data2, type = "response")

model2_predict <- factor(ifelse(model2_predict > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
model2_predict <- factor(model2_predict, levels = c("1", "0"))
selected_validation_data2$Purchase <- factor(selected_validation_data2$Purchase, levels = c("1", "0"))
cm_2 <- confusionMatrix(model2_predict, selected_validation_data2$Purchase)
cm_2

#--------------Model 3-------------#
#Significant variable set 2
selected_variables_3 <- c("source_a", "source_e", "source_h", "source_r", "source_u", "source_p", "source_w", 
                          "Frequency", "Days_since_Last_Update", "online_order", "Address_is_res", "Purchase")

model_3 <- glm(Purchase ~ source_a + source_e + source_h + source_r +
                 source_u + source_p +source_w + Frequency + 
                 Days_since_Last_Update + online_order + Address_is_res, data = train_data, family = binomial)

summary(model_3)

selected_validation_data3 <- validation_data[, selected_variables_3]
model3_predict <- predict(model_3, newdata = selected_validation_data3, type = "response")

model3_predict <- factor(ifelse(model3_predict > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
model3_predict <- factor(model3_predict, levels = c("1", "0"))
selected_validation_data3$Purchase <- factor(selected_validation_data3$Purchase, levels = c("1", "0"))
cm_3 <- confusionMatrix(model3_predict, selected_validation_data3$Purchase)
cm_3

#--------------Model 4-------------#
#Significance variable set 3
selected_variables_4 <- c("source_a", "source_h", "source_r", "source_u", "source_p", "source_w", 
                          "Frequency", "online_order", "Address_is_res", "Purchase")

model_4 <- glm(Purchase ~ source_a + source_h + source_r +
                 source_u + source_p +source_w + Frequency + 
                 online_order + Address_is_res, data = train_data, family = binomial)

summary(model_4)

selected_validation_data4 <- validation_data[, selected_variables_4]
model4_predict <- predict(model_4, newdata = selected_validation_data4, type = "response")

model4_predict <- factor(ifelse(model4_predict > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
model4_predict <- factor(model4_predict, levels = c("1", "0"))
selected_validation_data4$Purchase <- factor(selected_validation_data4$Purchase, levels = c("1", "0"))
cm_4 <- confusionMatrix(model4_predict, selected_validation_data4$Purchase)
cm_4

#--------------Model 5-------------#
#Significance variable set 4
selected_variables_5 <- c("source_a", "source_h", "source_r", "source_u", "source_w", 
                          "Frequency", "online_order", "Address_is_res", "Purchase")
model_5 <- glm(Purchase ~ source_a + source_h + source_r +
                 source_u +source_w + Frequency + 
                 online_order + Address_is_res, data = train_data, family = binomial)

summary(model_5)

selected_validation_data5 <- validation_data[, selected_variables_5]
model5_predict <- predict(model_5, newdata = selected_validation_data5, type = "response")

model5_predict <- factor(ifelse(model5_predict > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
model5_predict <- factor(model5_predict, levels = c("1", "0"))
selected_validation_data5$Purchase <- factor(selected_validation_data5$Purchase, levels = c("1", "0"))
cm_5 <- confusionMatrix(model5_predict, selected_validation_data5$Purchase)
cm_5
#--------------Model 6-------------#
#Significance variable set 5
selected_variables_6 <- c("source_a", "source_h", "source_u", "source_w", 
                          "Frequency", "online_order", "Address_is_res", "Purchase")

model_6 <- glm(Purchase ~ source_a + source_h +
                 source_u +source_w + Frequency + 
                 online_order + Address_is_res, data = train_data, family = binomial)

summary(model_6)

selected_validation_data6 <- validation_data[, selected_variables_6]
model6_predict <- predict(model_6, newdata = selected_validation_data6, type = "response")

model6_predict <- factor(ifelse(model6_predict > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
model6_predict <- factor(model6_predict, levels = c("1", "0"))
selected_validation_data6$Purchase <- factor(selected_validation_data6$Purchase, levels = c("1", "0"))
cm_6 <- confusionMatrix(model6_predict, selected_validation_data6$Purchase)
cm_6

#--------------Classification Tree-------------#
tree_model <- rpart(Purchase ~.-Spending, data = train_data, method = "class")
rpart.plot(tree_model)
validation_predictions_tree <- predict(tree_model, newdata = validation_data, type = "class")
validation_predictions_tree <- factor(validation_predictions_tree, levels = c("1", "0"))
validation_data$Purchase <- factor(validation_data$Purchase, levels = c("1", "0"))
conf_matrix_validation_tree <- confusionMatrix(validation_predictions_tree, validation_data$Purchase)
conf_matrix_validation_tree

#--------------PRUNING WITH BEST CP-------------#
best_cp1 <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
# Prune the tree with the best cp
pruned_tree1 <- prune(tree_model, cp = best_cp1)
# Plot the pruned tree
rpart.plot(pruned_tree1)

#------------Classification tree with important predictors from stepwise method----------#
selected_predictors_tree <- c("source_a", "source_e", "source_h", "source_r", "source_s", 
                              "source_t", "source_u", "source_p", "source_x", "source_w", 
                              "Frequency", "Days_since_Last_Update", "online_order", "Address_is_res", "Purchase")
train_data_subset_tree <- train_data[, selected_predictors_tree]
validation_data_subset_tree <- validation_data[, selected_predictors_tree]
tree_model_subset <- rpart(Purchase ~ ., data = train_data_subset_tree, method = "class")
rpart.plot(tree_model_subset)
validation_predictions_tree_subset <- predict(tree_model_subset, newdata = validation_data_subset_tree, type = "class")
validation_predictions_tree_subset <- factor(validation_predictions_tree_subset, levels = c("1", "0"))
validation_data$Purchase <- factor(validation_data$Purchase, levels = c("1", "0"))
conf_matrix_validation_tree_subset <- confusionMatrix(validation_predictions_tree_subset, validation_data$Purchase)
conf_matrix_validation_tree_subset

#--------------PRUNING WITH BEST CP few predictors-------------#
best_cp <- tree_model_subset$cptable[which.min(tree_model_subset$cptable[, "xerror"]), "CP"]
# Prune the tree with the best cp
pruned_tree <- prune(tree_model_subset, cp = best_cp)
# Plot the pruned tree
rpart.plot(pruned_tree)

#--------------KNN-------------#
#K= 3
library(caret)
train_data$Purchase <- factor(train_data$Purchase, levels = c("1", "0"))

#Train with knn=3
knn_model <- train(Purchase ~ ., data = train_data,
                   method = "knn",  
                   preProcess = c("center", "scale"),  # normalize data
                   tuneGrid = expand.grid(k = 3),
                   trControl = trainControl(method = "none"))

knn_predict <- predict(knn_model, newdata = validation_data)
knn_predict
validation_data$Purchase <- factor(validation_data$Purchase, levels = c("1", "0"))
cm_knn <- confusionMatrix(knn_predict, validation_data$Purchase)
cm_knn

#----Different K values----#
trControl <- trainControl(method = "LOOCV",  
                          allowParallel = TRUE)
knn_model_1 <- train(Purchase ~ ., data = train_data,
                     method = "knn",  
                     preProcess = c("center", "scale"), 
                     tuneGrid = expand.grid(k = seq(1, 3, 7)),  
                     trControl = trControl)
knn_model_1

knn_predict_1 <- predict(knn_model_1, newdata = validation_data)
validation_data$Purchase <- factor(validation_data$Purchase, levels = c("1","0"))
levels(validation_data$Purchase)
cm_knn_1 <- confusionMatrix(knn_predict_1, validation_data$Purchase)
cm_knn_1
best_k <- knn_model_1$bestTune$k
best_k

#--------------KNN using important predictors-------------#
selected_predictors <- c("source_a", "source_e", "source_h", "source_r", "source_s", 
                         "source_t", "source_u", "source_p", "source_x", "source_w", 
                         "Frequency", "Days_since_Last_Update", "online_order", "Address_is_res", "Purchase")

train_data_subset <- train_data[, selected_predictors]
validation_data_subset <- validation_data[, selected_predictors]
trControl <- trainControl(method = "LOOCV",  
                          allowParallel = TRUE)
knn_model <- train(Purchase ~ ., 
                   data = train_data_subset,
                   method = "knn",  
                   preProcess = c("center", "scale"), 
                   tuneGrid = expand.grid(k = seq(1, 3, 7)),  
                   trControl = trControl)
knn_predict <- predict(knn_model, newdata = validation_data_subset)
conf_matrix_knn <- confusionMatrix(knn_predict, validation_data$Purchase)
conf_matrix_knn
best_k_1 <- knn_model$bestTune$k
best_k_1

#****************************Using on Holdout data****************************#
#*#*************Classification best model****************#
#For initial logistic regression model
#Make predictions on the test data
initial_model_test <- predict(initial_model, newdata = test_data, type = "response")

#Convert predicted values to factors with correct levels
initial_model_test <- factor(ifelse(initial_model_test > 0.5, "1", "0"), levels = levels(factor(train_data$Purchase)))
initial_model_test <- factor(initial_model_test, levels = c("1", "0"))
test_data$Purchase <- factor(test_data$Purchase, levels = c("1","0"))
#Confusion matrix on the test data
conf_matrix_test_logistic <- confusionMatrix(initial_model_test, test_data$Purchase)
conf_matrix_test_logistic

#****************************GOAL 2- REGRESSION****************************#
#-----------------Predicting Spending when purchase = 1--------------------#
purchasers_df <- north_point_data[north_point_data$Purchase == 1, ]
set.seed(2024)
train_index <- sample(rownames(purchasers_df), size = 0.4*nrow(purchasers_df))
val_index <- sample(setdiff(rownames(purchasers_df), train_index), size = 0.35*nrow(purchasers_df))
test_index <- setdiff(rownames(purchasers_df), union(train_index, val_index))

#--------------Data partitioning----------------#
train_spending <- purchasers_df[train_index,]
validation_spending <- purchasers_df[val_index,]
test_spending <- purchasers_df[test_index,]
dim(train_spending)
dim(validation_spending)
dim(test_spending)

#--------------Multiple-Linear regression model----------------#
lmodel <- lm(Spending ~. -Purchase, data = train_spending)
summary(lmodel)
#Validate the model
validation_predictions <- predict(lmodel, newdata = validation_spending)
mae <- mean(abs(validation_predictions - validation_spending$Spending))
mae

#--------------Step-wise----------------#
stepwise_lmodel <- step(lmodel, direction = "backward")
#Linear regression after stepwise
selected_model <- lm(Spending ~ US + source_a + source_h + source_t + source_u + Frequency + 
                       Days_since_Last_Update + Address_is_res, data = train_spending)
# Validate the model
validation_predictions_1 <- predict(selected_model, newdata = validation_spending)
mae1 <- mean(abs(validation_predictions_1 - validation_spending$Spending))
mae1

#--------------Regression Tree----------------#
tree_model_spending <- rpart(Spending ~ . -Purchase, data = train_spending)
rpart.plot(tree_model_spending)
validation_predictions_tree <- predict(tree_model_spending, newdata = validation_spending)
mae_tree <- mean(abs(validation_predictions_tree - validation_spending$Spending))
mae_tree

#****************************Using on Holdout data****************************#
#For stepwise linear regression model
#Make predictions on the test data
lm_test <- predict(stepwise_lmodel, newdata = test_spending)
lm_test_mae <- mean(abs(lm_test - test_spending$Spending))
lm_test_mae

#****************************Estimated gross profit****************************#
#No. of customers = 180000
#Response rate = 0.053
#cost per mailing = 2
#Mean of spending when purchase is 1 = 205.249
Estimated_gross_profit <- 180000 * (0.053*mean(north_point_data$Spending[north_point_data$Purchase == 1])-2)
Estimated_gross_profit

#****************************Adding columns on test data****************************#
head(test_data)
dim(test_data)

#----------Predicted probability of purchase----------#
test_data$Predicted_Prob_Purchase <- predict(initial_model, newdata = test_data, type = "response")
head(test_data)

#----------Predicted Spending----------#
test_data$Predicted_Spending <- predict(selected_model, newdata = test_data)

#----------Adjusted probability of purchase----------#
purchase_rate <- 0.1065 # Original purchase rate
test_data$Adjusted_Prob_Purchase <- test_data$Predicted_Prob_Purchase * purchase_rate
head(test_data)

#----------Expected Spending----------#
test_data$Expected_Spending <- test_data$Predicted_Spending * test_data$Adjusted_Prob_Purchase
head(test_data)
sum(test_data$Expected_Spending)

#****************************Cumulative gain chart****************************#
expected_spending <- test_data$Expected_Spending
gain <- gains(test_data$Spending, expected_spending)
df <- data.frame(
  ncases = c(0, gain$cume.obs),
  cumExpectedSpending = c(0, gain$cume.pct.of.total * sum(expected_spending))
)

g1 <- ggplot(df, aes(x = ncases, y = cumExpectedSpending)) +
  geom_line() +
  geom_line(data = data.frame(ncases = c(0, nrow(test_data)), cumExpectedSpending = c(0, sum(expected_spending))),
            color = "gray", linetype = 2) + # adds baseline
  labs(x = "# Cases", y = "Cumulative Expected Spending", title = "Cumulative Expected Spending Chart") +
  scale_y_continuous(labels = scales::dollar_format())

#Decile-wise gain chart
df_decile <- data.frame(
  percentile = gain$depth,
  meanResponse = gain$mean.resp
)

g2 <- ggplot(df_decile, aes(x = percentile, y = meanResponse)) +
  geom_bar(stat = "identity") +
  labs(x = "Percentile", y = "Decile mean", title = "Decile-wise Expected Spending")

grid.arrange(g1, g2, ncol = 2)

#Test data in csv file
#write.csv(test_data, "new_test_data.csv", row.names = FALSE)

'#########################################################################################################################################
__________________________________________________________________________________________________________________________________

                           PROJECT  2: EMPOWERING USED MOBILE-DEVICE BUYERS   

__________________________________________________________________________________________________________________________________
##########################################################################################################################################'

#****************************COLLECTING DATA****************************#
used_device <- read.csv("used_device_data.csv")
str(used_device)
dim(used_device)

#****************************DATA EXPLORATION****************************#
#---Summary stats for numeric variables--#
numeric_data <- used_device[, sapply(used_device, is.numeric)]
summary(numeric_data)
#----- Check for missing values-----#
sum(is.na(used_device))
#Missing values in each column
colSums(is.na(used_device))
colnames(used_device)
md.pattern(used_device, rotate.names = TRUE)

#****************************HANDLING MISSING VALUES****************************#
#Group by device_brand and replace missing values with the median of each group
no_missing <- used_device %>%
  group_by(device_brand) %>%
  mutate_at(vars(rear_camera_mp, internal_memory, ram, battery, weight, front_camera_mp, screen_size), 
            funs(ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
  ungroup()

#Verify if missing values have been handled
colSums(is.na(no_missing))

no_missing <- no_missing[!is.na(no_missing$rear_camera_mp), ]
colSums(is.na(no_missing))

#****************************CHECK FOR 0 VALUES****************************#
zero_values <- colSums(used_device == 0, na.rm = TRUE)
zero_values

#****************************ATTRIBUTE ANALYSIS****************************#
#----------DISTRIBUTION OF TARGET VARIABLE-----------#
#Distribution of Used Price
data <- data.frame(normalized_used_price = no_missing$normalized_used_price)
#histogram with different colors for each bar
ggplot(data, aes(x = normalized_used_price)) +
  geom_histogram(aes(fill = ..x..), bins = 30) +
  scale_fill_gradient(low = "blue", high = "red") + 
  labs(title = "Distribution of Normalized Used Price")

#----------DISTRIBUTION OF CATEGORICAL-----------#
#-----OS-----#
#Distribution of OS
ggplot(no_missing, aes(x = os, fill = os)) +
  geom_bar(width = 0.5) +
  scale_fill_manual(values = rainbow(length(unique(no_missing$os)))) +  
  labs(title = "Distribution of Operating System") +
  geom_text(stat = 'count', aes(label = ..count..))

#-----BRAND-----#
ggplot(no_missing, aes(x = device_brand, fill = device_brand)) +
  geom_bar(width = 0.90) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, color = "black") +
  scale_fill_manual(values = rainbow(length(unique(no_missing$device_brand)))) +
  labs(title = "Count of Each Device Brand") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#-----NETWORK TYPE-----#
#4g 5g and other network contribution
#read the data in new dataframe
other_network <- read.csv("used_device_data.csv")
other_network$Others <- 0
other_network$X4g <- ifelse(other_network$X4g == "yes", 1, 0)
other_network$X5g <- ifelse(other_network$X5g == "yes", 1, 0)

other_network$Others <- ifelse(rowSums(other_network[, c("X4g", "X5g")]) == 0, 
                               1, other_network$Others)
yes_4g <- sum(other_network$X4g == 1)
yes_5g <- sum(other_network$X5g == 1)
yes_other <- sum(other_network$Others == 1)

#Combine frequencies of "yes" for each connectivity type into one table
combined_table <- c(yes_4g, yes_5g, yes_other)

# Create a bar plot for the combined frequencies
barplot(combined_table, 
        main = "Network Distribution", 
        names.arg = c("4G", "5G", "Other"), 
        xlab = "Connectivity Type", 
        ylab = "Frequency",
        col = "#00897B",
        las = 2,  # Rotate x-axis labels vertically
        ylim = c(0, max(combined_table) + 50))

#----------DISTRIBUTION OF NUMERICAL-----------#
#---Screen size----#
ggplot(no_missing, aes(x = screen_size)) +
  geom_histogram(fill = "darkseagreen1", color = "black") +
  labs(title = "Distribution of Screen Sizes")

#---Back Camera----#
ggplot(no_missing, aes(x = rear_camera_mp)) +
  geom_histogram(binwidth = 1, fill = "darkseagreen1", color = "black") + 
  labs(title = "Distribution of back camera size")  

#---Front Camera----#
ggplot(no_missing, aes(x = front_camera_mp)) +
  geom_histogram(fill = "darkseagreen1", color = "black") + 
  labs(title = "Distribution of front camera size")  

#---Internal Memory----#
ggplot(no_missing, aes(x = internal_memory)) +
  geom_histogram(fill = "darkseagreen1", color = "black") +  
  labs(title = "Distribution of internal memory")  

#---RAM----#
ggplot(no_missing, aes(x = ram)) +
  geom_histogram(binwidth = 1, fill = "darkseagreen1", color = "black") +  
  labs(title = "Distribution of ram") 

#---Battery----#
ggplot(no_missing, aes(x = battery)) +
  geom_histogram(fill = "darkseagreen1", color = "black") +  
  labs(title = "Distribution of battery") 

#---Weight----#
ggplot(no_missing, aes(x = weight)) +
  geom_histogram(fill = "darkseagreen1", color = "black") +  
  labs(title = "Distribution of weight") 

#---release year----#
ggplot(no_missing, aes(x = release_year)) +
  geom_histogram(binwidth = 1, fill = "darkseagreen1", color = "black") +  
  labs(title = "Distribution of release year") +
  scale_x_continuous(breaks = unique(no_missing$release_year))

#---Days used----#
ggplot(no_missing, aes(x = days_used)) +
  geom_histogram(fill = "darkseagreen1", color = "black") + 
  labs(title = "Distribution of days_used") 

#------------------BOX PLOTS-------------------#
#Reshape the data for box plot
melted_data <- melt(no_missing, measure.vars = c("screen_size", "rear_camera_mp", "front_camera_mp", 
                                                 "internal_memory", "ram", "battery", "weight", 
                                                 "release_year", "days_used", "normalized_used_price", 
                                                 "normalized_new_price"))

#Create box plots
ggplot(melted_data, aes(x = "", y = value)) +
  geom_boxplot(fill = "skyblue", color = "blue") +
  labs(title = "Boxplot of Numeric Attributes", x = NULL, y = NULL) +
  theme(axis.text = element_blank(),  
        strip.placement = "outside") +  
  facet_wrap(~variable, scales = "free_y") 

#****************************PREDICTOR ANALYSIS AND RELEVANCY*****************************#
#*
#*********************ANALYSIS*********************#

#-----Average Price by RAM-----# 
#Calculate average price for each RAM capacity
avg_price_by_ram <- no_missing %>%
  group_by(ram) %>%
  summarise(avg_price = mean(normalized_used_price))
#Plot average price by RAM capacity using a bar plot with different colors
ggplot(avg_price_by_ram, aes(x = factor(ram), y = avg_price, fill = factor(ram))) +
  geom_bar(stat = "identity") +
  labs(title = "Average Price by RAM Capacity", x = "RAM Capacity", y = "Average Price") +
  scale_fill_manual(values = rainbow(length(levels(factor(avg_price_by_ram$ram)))))

#---Relation between New price and Used price---#
ggplot(no_missing, aes(x = normalized_new_price, y = normalized_used_price)) +
  geom_point() +  # Add points
  labs(title = "Scatter Plot of Normalized Used Price vs. Normalized New Price",
       x = "Normalized New Price",
       y = "Normalized Used Price")

#---Correlations----#
numeric_variables <- no_missing %>% select_if(is.numeric)
# Calculate correlation matrix
correlation_matrix <- cor(numeric_variables)
corrplot(correlation_matrix)

#---Price by brand---#
ggplot(no_missing, aes(x = device_brand, y = normalized_used_price)) +
  geom_bar(stat = "summary", fun = "median", fill = "aquamarine3", color = "black") +  # Use bar plot with median as summary statistic
  labs(title = "Price by Brand", x = "Device Brand", y = "Normalized Used Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#---Price by OS---#
ggplot(no_missing, aes(x = os, y = normalized_used_price)) +
  geom_bar(stat = "summary", fun = "median", fill = "aquamarine3", color = "black") +  # Use bar plot with median as summary statistic
  labs(title = "Price by OS", x = "Operating System", y = "Normalized Used Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = rainbow(length(unique(no_missing$os))))

#*********************RELEVANCY**********************#

#-----------------FEATURE SELECTION-----------------#

#----LASSO METHOD----#
#Define response variable
y <- no_missing$normalized_used_price
#Remove the response variable from the dataset
data <- no_missing[, !names(no_missing) %in% c("normalized_used_price")]
head(data)
#Create design matrix
X <- model.matrix(~., data = data)
#Fit LASSO regression model using cross-validation
lasso_fs <- cv.glmnet(X, y, alpha = 1, family = "gaussian")
plot(lasso_fs)
#Get coefficients for the best lambda
best_lambda <- lasso_fs$lambda.min
lasso_coefficients <- coef(lasso_fs)
#Identify significant predictors with non-zero coefficients
lasso_significant_predictors <- rownames(lasso_coefficients)[apply(lasso_coefficients != 0, 1, any)]
#Print significant predictors
print(lasso_significant_predictors)

lasso_fs2 <- glmnet(X, y, alpha = 1, family = "gaussian")
plot(lasso_fs2, xvar = "lambda", label = TRUE)

#---BORUTA METHOD----#
set.seed(2024)
boruta_plot <- Boruta(normalized_used_price ~ ., data = no_missing)
boruta_plot

plot(boruta_plot, xlab = "", xaxt = "n")
lz <- lapply(1:ncol(boruta_plot$ImpHistory), function(i)
  boruta_plot$ImpHistory[is.finite(boruta_plot$ImpHistory[, i]), i])
names(lz) <- colnames(boruta_plot$ImpHistory)
lb <- sort(sapply(lz, median))
axis(side = 1, las = 2, labels = names(lb), at = 1:ncol(boruta_plot$ImpHistory), cex.axis = 0.5, font = 4)

#final_boruta_result <- TentativeRoughFix(boruta_plot)
#print(final_boruta_result)
#final_boruta_result$finalDecision

boruta_predictors <- getSelectedAttributes(boruta_plot, withTentative = FALSE)

#-----STEPWISE----#
set.seed(2024)
stepwise_model <- step(lm(normalized_used_price ~ ., data = no_missing))
summary(stepwise_model)
stepwise_predictors <- names(stepwise_model$coefficients)[!is.na(stepwise_model$coefficients)]
stepwise_predictors

#comparing the models
common_predictors <- Reduce(intersect, list(stepwise_predictors, boruta_predictors))
common_predictors

#****************************DATA TRANSFORMATION*****************************#
#---Column names changing---#
names(no_missing)[names(no_missing) == "device_brand"] <- "brand"
head(no_missing)

#****************************DIMENSION REDUCTION*****************************#
#------Combining 4G, 5G and Other netwroks into one column-------#
no_missing$network_type <- ifelse(no_missing$X5g == "yes", "5g",
                                  ifelse(no_missing$X4g == "yes", "4g", "Other"))
no_missing$network_type <- factor(no_missing$network_type, levels = c("4g", "5g", "Other"))
no_missing <- no_missing[, !(names(no_missing) %in% c("X4g", "X5g"))]
head(no_missing)

#Write csv file
#write.csv(no_missing, "Clean_Data.csv", row.names = FALSE)

############################################################################################
#*************************DATA PREPARATION FOR REGRESSION MODELS*************************#
#no_missing <- read.csv("Clean_Data.csv")
regression_data <- no_missing
colnames(regression_data)
colSums(is.na(regression_data))

#Convert into numeric
regression_data$os <- ifelse(regression_data$os == "Android", 1,
                             ifelse(regression_data$os == "iOS", 2,
                                    ifelse(regression_data$os == "Windows", 3, 4)))

regression_data$network_type <- ifelse(regression_data$network_type == "4g", 1,
                                       ifelse(regression_data$network_type == "5g", 2, 3))

#Create a vector of unique brands
unique_brands <- unique(regression_data$brand)
#Create a function to assign numeric values to each brand
assign_numeric <- function(brand) {
  return(match(brand, unique_brands))
}
#Apply the function to the brand column
regression_data$brand <- sapply(regression_data$brand, assign_numeric)

#*******************Tablet And Phone Division*******************#
tablet_dataset <- regression_data[regression_data$screen_size >= 18, ]
phone_dataset <- regression_data[regression_data$screen_size < 18, ]
colSums(is.na(tablet_dataset))
colSums(is.na(phone_dataset))

#******************Data Partitioning for the Tablet (Regression Models)******************#
set.seed(2024)
#Create the data partition
partition_index <- createDataPartition(tablet_dataset$normalized_used_price, p = 0.7, list = FALSE)
#Split the data into training and testing sets
train_tablet <- tablet_dataset[partition_index, ]
test_tablet <- tablet_dataset[-partition_index, ]
dim(train_tablet)
dim(test_tablet)

#******************Data Partitioning for the Phone (Regression Models)******************#
set.seed(2024)
#Create the data partition
partition_index_phone <- createDataPartition(phone_dataset$normalized_used_price, p = 0.7, list = FALSE)
#Split the data into training and testing sets
train_phone <- phone_dataset[partition_index_phone, ]
test_phone <- phone_dataset[-partition_index_phone, ]
dim(train_phone)
dim(test_phone)

#******************************REGRESSION MODELS FOR TABLET************************#
#------------------------------LINEAR REGRESSION MODEL(TABLETS)------------------------------#
#WITH ALL PREDICTORS
set.seed(2024)
lm_model_tab <- lm(normalized_used_price ~ ., data = train_tablet)
summary(lm_model_tab)
lm_pred_tab <- predict(lm_model_tab, newdata = test_tablet)
mae_lm_tab <- mean(abs(lm_pred_tab - test_tablet$normalized_used_price))
mae_lm_tab

#WITH SIGNIFICANT PREDICTORS
set.seed(2024)
sig_pred_model1_tab <- lm(normalized_used_price ~ brand + rear_camera_mp + front_camera_mp + internal_memory +
                            ram + battery + weight + normalized_new_price, data = train_tablet)
pred_sig_tab <- predict(sig_pred_model1_tab, newdata = test_tablet)
mae1_tab <- mean(abs(pred_sig_tab - test_tablet$normalized_used_price))
mae1_tab

#STEPWISE METHOD(BACKWARD)
set.seed(2024)
lm_backward_tab <- step(lm_model_tab, direction = "backward")
predictions_backward_tab <- predict(lm_backward_tab, newdata = test_tablet)
# Calculate MAE for the backward-selected model
mae_backward_tab <- mean(abs(predictions_backward_tab - test_tablet$normalized_used_price))
mae_backward_tab

#------------------------------REGRESSION TREE MODEL(TABLETS)------------------------------#
#TREE WITH ALL PREDICTORS
set.seed(2024)
tree_model_tab <- rpart(normalized_used_price ~ ., data = train_tablet, method = "anova")
rpart.plot(tree_model_tab)
fancyRpartPlot(tree_model_tab)
predictions_tree_tab <- predict(tree_model_tab, newdata = test_tablet)
mae_tree_tab <- mean(abs(predictions_tree_tab - test_tablet$normalized_used_price))
mae_tree_tab

#PRUNING WITH BEST CP
#Find the best complexity parameter (cp)
best_cp_tab <- tree_model_tab$cptable[which.min(tree_model_tab$cptable[, "xerror"]), "CP"]
#Prune the tree with the best cp
pruned_tree_tab <- prune(tree_model_tab, cp = best_cp_tab)
#Plot the pruned tree
rpart.plot(pruned_tree_tab)
fancyRpartPlot(pruned_tree_tab)

#TREE WITH SIGNIFICANT PREDICTORS
set.seed(2024)
tree_model_sig_tab <- rpart(normalized_used_price ~ brand + rear_camera_mp + front_camera_mp + internal_memory +
                              ram + battery + weight + normalized_new_price, data = train_tablet)
fancyRpartPlot(tree_model_sig_tab)
prediction_tree2_tab <- predict(tree_model_sig_tab, newdata = test_tablet)
mae_tree2_tab <- mean(abs(prediction_tree2_tab - test_tablet$normalized_used_price))
mae_tree2_tab

#******************************REGRESSION MODELS FOR PHONE************************#
#------------------------------LINEAR REGRESSION MODEL(PHONES)------------------------------#
#WITH ALL PREDICTORS
set.seed(2024)
lm_model_phone <- lm(normalized_used_price ~ ., data = train_phone)
summary(lm_model_phone)
# Generate predictions using the linear regression model
lm_pred_phone <- predict(lm_model_phone, newdata = test_phone)
mae_lm_phone <- mean(abs(lm_pred_phone - test_phone$normalized_used_price))
mae_lm_phone

#WITH SIGNIFICANT PREDICTORS
set.seed(2024)
sig_pred_model1_phone <- lm(normalized_used_price ~ os + screen_size + rear_camera_mp + front_camera_mp +
                              ram +  weight + release_year + normalized_new_price + network_type, data = train_phone)
pred_sig_phone <- predict(sig_pred_model1_phone, newdata = test_phone)
mae1_phone <- mean(abs(pred_sig_phone - test_phone$normalized_used_price))
mae1_phone

#STEPWISE METHOD (BACKWARD)
set.seed(2024)
lm_backward_phone <- step(lm_model_phone, direction = "backward")
predictions_backward_phone <- predict(lm_backward_phone, newdata = test_phone)
# Calculate MAE for the backward-selected model
mae_backward_phone <- mean(abs(predictions_backward_phone - test_phone$normalized_used_price))
mae_backward_phone

#------------------------------REGRESSION TREE MODEL(PHONES)------------------------------#
#TREE WITH ALL PREDICTORS
set.seed(2024)
tree_model_phone <- rpart(normalized_used_price ~ ., data = train_phone, method = "anova")
rpart.plot(tree_model_phone)
fancyRpartPlot(tree_model_phone)
predictions_tree_phone <- predict(tree_model_phone, newdata = test_phone)
mae_tree_phone <- mean(abs(predictions_tree_phone - test_phone$normalized_used_price))
mae_tree_phone

#PRUNING WITH BEST CP
best_cp_phone <- tree_model_phone$cptable[which.min(tree_model_phone$cptable[, "xerror"]), "CP"]
#Prune the tree with the best cp
pruned_tree_phone <- prune(tree_model_phone, cp = best_cp_phone)
rpart.plot(pruned_tree_phone)
fancyRpartPlot(pruned_tree_phone)

#TREE WITH IMPORTANT PREDICTORS
set.seed(2024)
tree_model_sig_phone <- rpart(normalized_used_price ~ os + screen_size + rear_camera_mp + front_camera_mp +
                                ram +  weight + release_year + normalized_new_price + network_type, data = train_phone)
fancyRpartPlot(tree_model_sig_phone)
prediction_tree2_phone <- predict(tree_model_sig_phone, newdata = test_phone)
mae_tree2_phone <- mean(abs(prediction_tree2_phone - test_phone$normalized_used_price))
mae_tree2_phone

#***********************SELECTED REGRESSION MODEL FOR TABLET***********************#
#******Classifying the predicted price into low, medium and high*******#
predictions_final_tab <- data.frame(Predicted_Price = pred_sig_tab)
#Combine the predictions with the original test data
predicted_data_tab <- cbind(test_tablet, predictions_final_tab)
colnames(predicted_data_tab)

#Classifying the predicted price into low, medium and high for tablet dataset.
predicted_data_tab$Price_Category <- cut(predicted_data_tab$Predicted_Price, 
                                         breaks = c(4.10, 5.80, 6.10, Inf), 
                                         labels = c("Low", "Medium", "High"), 
                                         include.lowest = TRUE)
#Write to a CSV file
#write.csv(predicted_data_tab, "predicted_prices_tablet.csv", row.names = FALSE)

#***********************SELECTED REGRESSION MODEL FOR PHONE***********************#
predictions_final_phone <- data.frame(Predicted_Price = pred_sig_phone)
#Combine the predictions with the original test data
predicted_data_phone <- cbind(test_phone, predictions_final_phone)
colnames(predicted_data_phone)

#Classifying the predicted price into low, medium and high for phone dataset.
predicted_data_phone$Price_Category <- cut(predicted_data_phone$Predicted_Price, 
                                           breaks = c(2.6, 4.0, 5.7, Inf), 
                                           labels = c("Low", "Medium", "High"), 
                                           include.lowest = TRUE)
# Write to a CSV file
#write.csv(predicted_data_phone, "predicted_prices_phone.csv", row.names = FALSE)

###########################--------PHASE 2--------###########################
#               CLASSIFICATION ALGORITHMS AS PER PROJECT REQUIREMENT

#As per project requirement performing classification algorithms also but will consider classifying the predicted price for this project.

#***********************DATA PREPARATION FOR CLASSIFICATION MODELS***********************#
classification_data <- regression_data

#----Group used price into 2 categories(low, high)----#
used_price_categories <- integer(nrow(classification_data))
# Iterate over each row
for (i in 1:nrow(classification_data)) {
  if (classification_data$normalized_used_price[i] < 4.0) {
    used_price_categories[i] <- "low"
  } else {
    used_price_categories[i] <- "high"
  }
}

table(used_price_categories)
classification_data$used_price_category <- as.factor(used_price_categories)
#----Removing columns that are not needed for classification
classification_data <- classification_data[, -c(12,13)]
head(classification_data)
str(classification_data)
#*******************Tablet And Phone Division (classification model)********************#
tablet_dataset_class <- classification_data[classification_data$screen_size >= 18, ]
phone_dataset_class <- classification_data[classification_data$screen_size < 18, ]
colSums(is.na(tablet_dataset_class))
colSums(is.na(phone_dataset_class))
dim(tablet_dataset_class)
dim(phone_dataset_class)

#******************Data Partitioning for the Tablet (Classification Models)******************#
set.seed(2024)
# Create the data partition
partition_index_class <- createDataPartition(tablet_dataset_class$used_price_category, p = 0.7, list = FALSE)
# Split the data into training and testing sets
train_tablet_class <- tablet_dataset_class[partition_index_class, ]
test_tablet_class <- tablet_dataset_class[-partition_index_class, ]
dim(train_tablet_class)
dim(test_tablet_class)

table(train_tablet_class$used_price_category)
table(test_tablet_class$used_price_category)

#******************Data Partitioning for the Phone (Classification Models)******************#
set.seed(2024)
# Create the data partition
partition_index_phone1 <- createDataPartition(phone_dataset_class$used_price_category, p = 0.7, list = FALSE)
# Split the data into training and testing sets
train_phone_class <- phone_dataset_class[partition_index_phone1, ]
test_phone_class <- phone_dataset_class[-partition_index_phone1, ]
dim(train_phone_class)
dim(test_phone_class)

table(train_phone_class$used_price_category)
table(test_phone_class$used_price_category)

#******************************CLASSIFICATION MODELS FOR TABLETS************************#

#---------------------LOGISTIC REGRESSION MODEL--------------------#
logistic_model_tablet <- glm(used_price_category ~ ., data = train_tablet_class, family = "binomial")
predictions_tablet <- predict(logistic_model_tablet, newdata = test_tablet_class, type = "response")
predicted_classes_tablet <- ifelse(predictions_tablet > 0.5, "high", "low")
predicted_classes_tablet <- factor(predicted_classes_tablet, levels = levels(test_tablet_class$used_price_category))
confusion_matrix_tablet <- confusionMatrix(predicted_classes_tablet, test_tablet_class$used_price_category)
confusion_matrix_tablet

#---------------------KNN MODEL--------------------#
set.seed(2024)
knn_model_tab <- train(used_price_category ~ ., data = train_tablet_class,
                       method = "knn",  
                       preProcess = c("center", "scale"), 
                       tuneGrid = expand.grid(k = 3),
                       trControl = trainControl(method = "none"))

knn_predict_tab <- predict(knn_model_tab, newdata = test_tablet_class)
knn_predict_tab
cm_knn_tab <- confusionMatrix(knn_predict_tab, test_tablet_class$used_price_category)
cm_knn_tab

#******************************CLASSIFICATION MODELS FOR PHONES************************#

#---------------------LOGISTIC REGRESSION MODEL--------------------#
logistic_model_phone <- glm(used_price_category ~ ., data = train_phone_class, family = "binomial")
predictions_phone <- predict(logistic_model_phone, newdata = test_phone_class, type = "response")
predicted_classes_phone <- ifelse(predictions_phone > 0.5, "high", "low")
predicted_classes_phone <- factor(predicted_classes_phone, levels = levels(test_phone_class$used_price_category))
confusion_matrix_phone <- confusionMatrix(predicted_classes_phone, test_phone_class$used_price_category)
confusion_matrix_phone

#---------------------CLASSIFICATION TREE MODEL--------------------#
set.seed(2024)
classification_tree_phone <- rpart(used_price_category ~ ., data = train_phone_class, method = "class")
fancyRpartPlot(classification_tree_phone)
rpart.plot(classification_tree_phone)
str(classification_tree_phone)

predict_tree_tab <- predict(classification_tree_phone, newdata = test_phone_class, type = "class")
conf_matrix_tree <- confusionMatrix(predict_tree_tab, test_phone_class$used_price_category)
conf_matrix_tree

#PRUNING WITH BEST CP
best_cp_class_phone <- classification_tree_phone$cptable[which.min(classification_tree_phone$cptable[, "xerror"]), "CP"]
# Prune the tree with the best cp
pruned_tree_class_phone <- prune(classification_tree_phone, cp = best_cp_class_phone)
# Plot the pruned tree
rpart.plot(pruned_tree_class_phone)
fancyRpartPlot(pruned_tree_class_phone)

#---------------------KNN MODEL--------------------#
set.seed(2024)
knn_model_phone <- train(used_price_category ~ ., data = train_phone_class,
                         method = "knn",  
                         preProcess = c("center", "scale"), 
                         tuneGrid = expand.grid(k = 3),
                         trControl = trainControl(method = "none"))

knn_predict_phone <- predict(knn_model_phone, newdata = test_phone_class)
knn_predict_phone
cm_knn_phone <- confusionMatrix(knn_predict_phone, test_phone_class$used_price_category)
cm_knn_phone

#--------NEW AND USED PRICE COMPARISION FOR PHONES AND TABLETS-------------#
#Calculate total new and used prices for tablets
tablet_new_price_total <- sum(tablet_dataset$normalized_new_price, na.rm = TRUE)
tablet_used_price_total <- sum(tablet_dataset$normalized_used_price, na.rm = TRUE)
#Calculate total new and used prices for phones
phone_new_price_total <- sum(phone_dataset$normalized_new_price, na.rm = TRUE)
phone_used_price_total <- sum(phone_dataset$normalized_used_price, na.rm = TRUE)

#Create a data frame for the tablet pie chart
tablet_pie_data <- data.frame(Category = c("New", "Used"),
                              Total_Price = c(tablet_new_price_total, tablet_used_price_total))
#Create a data frame for the phone pie chart
phone_pie_data <- data.frame(Category = c("New", "Used"),
                             Total_Price = c(phone_new_price_total, phone_used_price_total))

#Create the pie chart for tablets with percentages
tablet_pie <- ggplot(tablet_pie_data, aes(x = "", y = Total_Price, fill = Category)) +
  geom_bar(stat = "identity", width = 1) +
  geom_text(aes(label = scales::percent(Total_Price/sum(Total_Price))), position = position_stack(vjust = 0.5)) +
  coord_polar("y", start = 0) +
  labs(title = "Tablet Price Comparison",
       fill = "Category") +
  theme_void() +
  theme(legend.position = "right",
        plot.title = element_text(hjust = 0.5))

#Create the pie chart for phones with percentages
phone_pie <- ggplot(phone_pie_data, aes(x = "", y = Total_Price, fill = Category)) +
  geom_bar(stat = "identity", width = 1) +
  geom_text(aes(label = scales::percent(Total_Price/sum(Total_Price))), position = position_stack(vjust = 0.5)) +
  coord_polar("y", start = 0) +
  labs(title = "Phone Price Comparison",
       fill = "Category") +
  theme_void() +
  theme(legend.position = "right",
        plot.title = element_text(hjust = 0.5))

#Arrange the pie charts in a grid
grid.arrange(tablet_pie, phone_pie, nrow = 1)


'#########################################################################################################################################
__________________________________________________________________________________________________________________________________

                      PROJECT  3: MORTGAGE PAYBACK BEHAVIOR ANALYTICS   

__________________________________________________________________________________________________________________________________
#########################################################################################################################################'

#****************************COLLECTING DATA****************************#
loan_data <- read.csv("Mortgage.csv")
str(loan_data)
dim(loan_data)

#****************************CHECKING MISSING VALUES****************************#
colSums(is.na(loan_data))
#Visual representation of the missing values
#This takes 2 minutes of time to run so keeping it in comments. Can run and check when needed
#md.pattern(loan_data, rotate.names = TRUE) #270 missing values in LTV_time

#**************************** MISSING VALUES in CSV FILE****************************#
missing_ltv <- loan_data[is.na(loan_data$LTV_time), ]
#write.csv(missing_ltv, file = "missing_ltv_records.csv", row.names = FALSE)
#Check how many IDs have missing values
length(unique(missing_ltv$id))
#18 borrowers have missing value

#****************************HANDLING MISSING VALUES****************************#
#Removing the missing values for now
loan_data <- na.omit(loan_data)
#write.csv(loan_data, file = "Clean_Data.csv", row.names = FALSE)
#Check how many IDs are there after eliminating the missing records
length(unique(loan_data$id))
#Out of 50,000 borrower IDs there are 49982 after removing the missing records

#****************************CHECKING 0 VALUES****************************#
colSums(loan_data == 0, na.rm = TRUE)
#Find IDs with zero interest rate
zero_interest_time_ids <- loan_data$id[loan_data$interest_rate_time == 0]
#Find IDs with zero interest rate at origination time
zero_interest_orig_ids <- loan_data$id[loan_data$Interest_Rate_orig_time == 0]

#****************************HANDLING 0 VALUES****************************#
#Replace 0 values in interest_rate_time by taking median for each ID
loan_data <- loan_data %>%
  group_by(id) %>%
  mutate(interest_rate_time = ifelse(interest_rate_time == 0, median(interest_rate_time, na.rm = TRUE), interest_rate_time))
#Checking IDs with still 0 Values in interest rate
zero_interest_time_ids <- loan_data$id[loan_data$interest_rate_time == 0]
zero_interest_time_ids
#Removing Ids that still have 0 values in interest rate
loan_data <- loan_data[!(loan_data$id %in% zero_interest_time_ids), ] # 2 IDs are removed (Total 49980 Ids present)

#Replace 0 values in Interest_Rate_orig_time by taking median for each ID
loan_data <- loan_data %>%
  group_by(id) %>%
  mutate(Interest_Rate_orig_time = ifelse(Interest_Rate_orig_time == 0, median(Interest_Rate_orig_time, na.rm = TRUE), Interest_Rate_orig_time))
#Check for 0s again
colSums(loan_data == 0, na.rm = TRUE)
#there a a lot of 0s in Interest_Rate_orig_time. So to handle it, group by id- take median of interest rate of each id. Replace with respected missing IDs
median_interest_rate <- loan_data %>%
  group_by(id) %>%
  summarise(median_interest_rate = median(interest_rate_time, na.rm = TRUE))
loan_data <- loan_data %>%
  left_join(median_interest_rate, by = "id") %>%
  mutate(Interest_Rate_orig_time = ifelse(Interest_Rate_orig_time == 0, median_interest_rate, Interest_Rate_orig_time)) %>%
  select(-median_interest_rate)
colSums(loan_data == 0, na.rm = TRUE)

#****************************DATA PREPROCESSING****************************#
#---------To remove IDs with only one record for borrowers--------#
#No. of records for each borrower
id_record_counts <- loan_data %>%
  group_by(id) %>%
  summarise(num_records = n())
#Removing IDs with just one observation
id_record_counts_filtered <- id_record_counts %>%
  filter(num_records < 2)
ids_to_remove <- id_record_counts_filtered$id
# Remove IDs with only one record from loan_data
loan_data <- loan_data %>%
  filter(!(id %in% ids_to_remove))

#---------------------For better pre-processing sql is being used---------------------#

#Get the data into seperate dataframe for performing SQL queries
loan_sql <- loan_data
colnames(loan_sql)

#Transforming time column by capturing only the no. of observations each ID has 
observation_count <- sqldf('SELECT id, count(id) AS observation_count FROM loan_sql GROUP BY id')

#Transforming orig_time column
orig_time <- sqldf('SELECT id, MAX(orig_time) AS origin_time FROM loan_sql GROUP BY id')

#Transforming first_time column
first_time <- sqldf('SELECT id, MAX(first_time) AS first_time FROM loan_sql GROUP BY id')

#Transforming balance_time column
last_balance_time <- loan_data %>%
  group_by(id) %>%
  summarise(balance_time = round(max(balance_time[which.max(time)])))
#Adding a new column
balance_due_times <- loan_data %>%
  group_by(id) %>%
  summarise(zero_diff_count = sum(diff(balance_time) == 0))

#Transforming LTV_time column
ltv_ratio <- sqldf('SELECT id, AVG(LTV_time) AS ltv_ratio FROM loan_sql GROUP BY id')

#Transforming interest_rate_time column
interest <- sqldf('SELECT id, AVG(interest_rate_time) AS interest_rate FROM loan_sql GROUP BY id')
#Adding new column fixed_interest rate 
fixedinterest <- sqldf('SELECT id, CASE 
        WHEN MIN(interest_rate_time) = MAX(interest_rate_time) THEN 1
        ELSE 0 END AS fixed_interest FROM loan_sql GROUP BY id')

#Transforming hpi_time column
hpi_index <- sqldf('SELECT id, hpi_time
FROM loan_sql
WHERE (id, time) IN (
        SELECT id,
            MAX(time) AS hpi_index
        FROM loan_sql GROUP BY id)')

#Transforming gdp_time, and uer_time columns
gdp_uer <- sqldf('SELECT id, gdp_time, uer_time FROM loan_sql
WHERE (id, time) IN (SELECT id, MAX(time) AS max_time FROM loan_sql GROUP BY id)')

#Transforming real estate type columns
realestate <- sqldf('SELECT id, CASE 
        WHEN REtype_CO_orig_time = 1 THEN 3
        WHEN REtype_PU_orig_time = 1 THEN 2
        WHEN REtype_SF_orig_time = 1 THEN 1
        ELSE 4
    END AS real_estate_type FROM loan_sql
WHERE (id, time) IN (SELECT id, MAX(time) AS real_estate_type FROM loan_sql GROUP BY id)')

#Transforming investor type columns
investor <- sqldf('SELECT id, investor_orig_time FROM loan_sql
WHERE (id, time) IN (SELECT id, MAX(time) AS investor FROM loan_sql GROUP BY id)')

#Transforming all origination time columns
origination_time_columns <- sqldf('SELECT id, balance_orig_time, FICO_orig_time, LTV_orig_time, Interest_Rate_orig_time, hpi_orig_time
FROM loan_sql
WHERE (id, time) IN (SELECT id, MAX(time) AS max_time FROM loan_sql GROUP BY id)')

#Status time
status <- sqldf('SELECT id, status_time
FROM loan_sql
WHERE (id, time) IN (SELECT id, MAX(time) AS max_time FROM loan_sql GROUP BY id)')

final_loan_data <- Reduce(function(x, y) merge(x, y, by = "id", all = TRUE),
                          list(observation_count, orig_time, first_time,last_balance_time, balance_due_times, ltv_ratio,
                               interest, fixedinterest, hpi_index, gdp_uer, realestate, 
                               investor, origination_time_columns, status))
colSums(is.na(final_loan_data))
colSums(final_loan_data == 0, na.rm = TRUE)

#****************************DIMENSION REDUCTION****************************#
duplicates <- final_loan_data[duplicated(final_loan_data$id), "id"]
table(duplicates)
final_loan_data_unique <- distinct(final_loan_data, id, .keep_all = TRUE)
colnames(final_loan_data_unique)
#write.csv(final_loan_data_unique, "Loan_Dataset.csv", row.names = FALSE)

#****************************DEFAULT AND PAYOFF DATAFRAMES****************************#
#--------------Dataframe of loan status default where last timestamp is considered-------------#
defaulters_data <- final_loan_data_unique %>%
  filter(status_time == 1)

#--------------Dataframe of loan status payoff where last timestamp is considered--------------#
payoffs_data <- final_loan_data_unique %>%
  filter(status_time == 2)

#****************************PREDICTOR ANALYSIS****************************#
#*********REAL ESTATE DISTRIBUTION************#
ggplot(final_loan_data_unique, aes(x = factor(real_estate_type))) +
  geom_bar(fill = rainbow(length(unique(final_loan_data_unique$real_estate_type))), color = "black") +
  labs(title = "Distribution of Real Estate Types", x = "Real Estate Type", y = "Frequency") +
  theme_minimal()

#*********ANALYZING ATTRIBUTE INFLUENCE ON DEFAULT AND PAYOF************#
#----------INVESTORS STATUS----------#
#Default type
p3 <- ggplot(defaulters_data, aes(x = factor(investor_orig_time))) + 
  geom_bar(fill = rainbow(length(unique(defaulters_data$investor_orig_time)))) + 
  labs(title = "Distribution of Investors Among Defaulters")
#Payoff type
p4 <- ggplot(payoffs_data, aes(x = factor(investor_orig_time))) + 
  geom_bar(fill = rainbow(length(unique(payoffs_data$investor_orig_time)))) + 
  labs(title = "Distribution of Investors Among Payoffs")
grid.arrange(p3, p4, ncol = 2)

#*********LOAN ORIGINATION ATTRIBUTES************#
#----------RELATION BETWEEN ORIGIN TIME (DEFAULT AND PAYOFF)----------#
p5 <- ggplot(defaulters_data, aes(x = origin_time, fill = factor(status_time))) +
  geom_bar(position = "stack", color = "black") +
  labs(title = "Origination Time and Defaulters",
       x = "Origination Time", y = "Count") +
  theme_minimal()
p6 <- ggplot(payoffs_data, aes(x = origin_time, fill = factor(status_time))) +
  geom_bar(position = "stack", color = "black") +
  labs(title = "Origination Time and Payoffs",
       x = "Origination Time", y = "Count") +
  theme_minimal()
grid.arrange(p5, p6, ncol = 2)

#----------FICO AMONG LOAN STATUS----------#
ggplot(defaulters_data, aes(x = FICO_orig_time)) +
  geom_histogram(fill = "violetred4", color = "black", bins = 50) +
  labs(title = "Distribution of FICO Scores among Defaulters",
       x = "FICO Score",
       y = "Frequency") +
  theme_minimal()
ggplot(payoffs_data, aes(x = FICO_orig_time)) +
  geom_histogram(fill = "violetred4", color = "black", bins = 50) +
  labs(title = "Distribution of FICO Scores among Payoffs",
       x = "FICO Score",
       y = "Frequency") +
  theme_minimal()

#----------DISTRIBUTION OF LTV AND LTV ORIGIN----------#
p7 <- ggplot(final_loan_data_unique, aes(x = ltv_ratio)) +
  geom_histogram(fill = "violetred4", color = "black", bins = 50) +
  labs(title = "Distribution of LTV",
       x = "Loan-to-Value (LTV) Ratio",
       y = "Frequency") +
  theme_minimal()
p8 <- ggplot(final_loan_data_unique, aes(x = LTV_orig_time)) +
  geom_histogram(fill = "violetred4", color = "black", bins = 50) +
  labs(title = "Distribution of LTV origin",
       x = "Loan-to-Value (LTV) Ratio",
       y = "Frequency") +
  theme_minimal()
grid.arrange(p7, p8, ncol = 2)

#*********RELATION BETWEEN GDP AND UNEMPLOYMENT RATE************#
#This code will take 2-3 min of time to run
ggplot(final_loan_data_unique, aes(x = gdp_time, y = uer_time)) +
  geom_point(color = "blue") +  
  labs(x = "GDP", y = "Unemployment Rate", title = "Scatter Plot: GDP vs Unemployment Rate") +  
  theme_minimal()

#*********FIXED INTEREST RATE WITH STATUS TIME************#
count1 <- sum(final_loan_data_unique$fixed_interest == 1 & final_loan_data_unique$status_time == 1)
count2<- sum(final_loan_data_unique$fixed_interest == 1 & final_loan_data_unique$status_time == 2)
counts <- data.frame(
  category = c("Fixed Interest = 1, Status Time = 1", "Fixed Interest = 0, Status Time = 2"),
  count = c(count1, count2)
)
x_labels <- ifelse(counts$category == "Fixed Interest = 1, Status Time = 1", "Default", "Paid-off")
barplot(counts$count, names.arg = x_labels, col = rainbow(length(counts$count)),
        main = "Counts of Fixed Interest and Status Time",
        xlab = "Category", ylab = "Count")

#****************************PREDICTOR RELEVANCY****************************#
#*CORRELATION
numeric_data <- final_loan_data_unique[, -c(9, 13, 14,20)]
correlation_matrix <- cor(numeric_data)
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", tl.col = "black")
correlation_matrix_status <- cor(final_loan_data_unique[, -c(1, 2, 8, 12)])  

#****************************DATA TRANSFORMATION****************************#
names(final_loan_data_unique)[names(final_loan_data_unique) == "observation_count"] <- "no_of_payments"
names(final_loan_data_unique)[names(final_loan_data_unique) == "zero_diff_count"] <- "missed_payment_times"
names(final_loan_data_unique)[names(final_loan_data_unique) == "first_time"] <- "first_payment_time"
names(final_loan_data_unique)[names(final_loan_data_unique) == "investor_orig_time"] <- "investor"
names(final_loan_data_unique)[names(final_loan_data_unique) == "balance_orig_time"] <- "balance_at_origin"
names(final_loan_data_unique)[names(final_loan_data_unique) == "FICO_orig_time"] <- "credit_score"
names(final_loan_data_unique)[names(final_loan_data_unique) == "LTV_orig_time"] <- "ltv_origin"
names(final_loan_data_unique)[names(final_loan_data_unique) == "Interest_Rate_orig_time"] <- "interest_rate_origin"
names(final_loan_data_unique)[names(final_loan_data_unique) == "hpi_orig_time"] <- "hpi_origin"
names(final_loan_data_unique)[names(final_loan_data_unique) == "status_time"] <- "status_type"
names(final_loan_data_unique)[names(final_loan_data_unique) == "hpi_time"] <- "hpi_index"
head(final_loan_data_unique)

#********************************SEPERATE DATASETS**********************************#
ongoing_data <- final_loan_data_unique[final_loan_data_unique$status_type == 0, ]
default_payoff_data <- final_loan_data_unique[final_loan_data_unique$status_type %in% c(1, 2), ]

#********************************STATUS TYPE TO 1 AND 0**********************************#
default_payoff_data$status_type <- ifelse(default_payoff_data$status_type == 1, 1, 0)
table(default_payoff_data$status_type)
#status type = 1 is defaulf and 0 is payoff

#********************************DATA PARTITIONING FOR DEFUALT AND PAYOFF STATUS**********************************#
set.seed(2024)
partition_index <- createDataPartition(default_payoff_data$status_type, p = 0.7, list = FALSE)
#Split the data into training and testing sets
train_default_payoff <- default_payoff_data[partition_index, ]
validation_default_payoff <- default_payoff_data[-partition_index, ]
dim(train_default_payoff)
dim(validation_default_payoff)
table(train_default_payoff$status_type)
table(validation_default_payoff$status_type)

#********************************CLASSIFICATION MODELS FOR STATUS TIME=1,2**********************************#
#--------LOGISTIC REGRESSION--------# 
logistic_model <- glm(status_type ~ ., data = train_default_payoff, family = "binomial")
predictions_glm <- predict(logistic_model, newdata = validation_default_payoff, type = "response")
predicted_classes_glm <- ifelse(predictions_glm > 0.5, 1, 0)
glm_conf_matrix <- confusionMatrix(factor(predicted_classes_glm), factor(validation_default_payoff$status_type))
glm_conf_matrix

#--------CLASSIFICATION TREE--------#
tree_model<- rpart(status_type ~ ., data = train_default_payoff, method = "class")
rpart.plot(tree_model)
fancyRpartPlot(tree_model)
tree_predictions <- predict(tree_model, validation_default_payoff, type = "class")
tree_conf_matrix <- confusionMatrix(factor(tree_predictions), factor(validation_default_payoff$status_type))
tree_conf_matrix

#--------KNN--------#
#Train with knn=3
train_default_payoff$status_type <- factor(train_default_payoff$status_type, levels = c("1", "0"))
train_default_payoff$status_type <- as.factor(train_default_payoff$status_type)
knn_model <- train(status_type ~ ., data = train_default_payoff,
                   method = "knn",  
                   preProcess = c("center", "scale"),  
                   tuneGrid = expand.grid(k = 3),
                   trControl = trainControl(method = "none"))
knn_predict <- predict(knn_model, newdata = validation_default_payoff)
knn_conf_matrix <- confusionMatrix(factor(knn_predict), factor(validation_default_payoff$status_type))
knn_conf_matrix

#--------NAIVE BAYES --------#
naive_bayes_model <- naiveBayes(status_type ~ ., data = train_default_payoff)
naive_bayes_predictions <- predict(naive_bayes_model, newdata = validation_default_payoff)
naive_bayes_conf_matrix <- confusionMatrix(factor(naive_bayes_predictions), factor(validation_default_payoff$status_type))
naive_bayes_conf_matrix

#********************************PREDICTING THE STATUS ON ONGOING DATA**********************************#
ongoing_data$prediction <- predict(logistic_model, newdata = ongoing_data, type = "response")
ongoing_data$prediction <- ifelse(ongoing_data$prediction > 0.5, 1, 0)
table(ongoing_data$prediction)

'__________________________________________________________________________________________________________________________________
____________________________________________________________________END______________________________________________________________'