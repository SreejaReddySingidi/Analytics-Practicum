###########################   PROJECT  2: USED DEVICE PRICE PREDICTIONS   #####################
#Name: Sreeja Reddy Singidi
#Last updated : 05/08/2024

#Libraries required
library(dplyr)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(glmnet)
library(Metrics)
library(rpart)
library(caret)
library(Boruta)
library(randomForest)
library(rpart.plot)
library(rattle)
library(neuralnet)
library(mice)
library(corrplot)
library(e1071)
library(class)

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

