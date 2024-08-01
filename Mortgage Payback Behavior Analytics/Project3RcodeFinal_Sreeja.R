###########################   PROJECT  3: MORTGAGE PAYBACK ANALYSIS   #####################

#Name: Sreeja Reddy Singidi
#Last updated : 05/08/2024

#Libraries required
library(mice)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(e1071)
library(sqldf)
library(corrplot)

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
