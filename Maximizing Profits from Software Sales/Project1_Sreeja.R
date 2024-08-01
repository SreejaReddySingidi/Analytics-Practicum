###########################   PROJECT  1: NORTH POINT SOFTWARE PRODUCTION COMPANY   #####################

#Name: Sreeja Reddy Singidi
#Last updated : 05/08/2024

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



