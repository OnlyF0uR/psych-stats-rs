if(!"readr" %in% installed.packages()) { install.packages("readr") }
if(!"dplyr" %in% installed.packages()) { install.packages("dplyr") }
if(!"ggplot2" %in% installed.packages()) { install.packages("ggplot2") }
if(!"car" %in% installed.packages()) { install.packages("car") }

# Load necessary libraries
library(readr)  # For reading CSV files
library(dplyr)  # For data manipulation
library(ggplot2)  # For visualization
library(car)  # For ANCOVA

# Read the data from CSV file
data <- read_csv("samples/data3.csv")

# Display the first few rows of the dataset
head(data)

# Ensure 'condition' is a factor
data$condition <- as.factor(data$condition)

# Perform ANCOVA
ancova_result <- aov(score ~ condition + age + happiness, data = data)
summary(ancova_result)

# Check assumptions: homogeneity of variance
leveneTest(score ~ condition, data = data)

# Post-hoc tests if necessary (e.g., Tukey's HSD)
posthoc <- TukeyHSD(ancova_result, "condition")
print(posthoc)

# Visualize the results
ggplot(data, aes(x = condition, y = score)) +
  geom_boxplot() +
  geom_jitter(aes(color = condition), width = 0.2, alpha = 0.5) +
  labs(title = "ANCOVA: Effect of Condition on Score",
       x = "Condition",
       y = "Score") +
  theme_minimal()