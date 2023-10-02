### The script includes three predictive models of human decision-making.
### The models are defined within an evidence accumulation framework and are obtained through the Beehive experiment.

### Libraries
library(rstudioapi)
library(ggplot2)
library(grid)
library(gridExtra)
library(effects)
library(dplyr)
library(scales)
library(psych)
library(lme4)
library(performance)
library(car)
library(ggeffects)
library(lmerTest)

# Directory
curdir <- getwd()
setwd(curdir)

### Reading the data
full_df <- data.frame()
n_pp <- 72

for (i in 1:n_pp){
  
  file_name <- paste('./DotsTask_sub',i,'.csv',sep = "")
  data <- read.csv(file_name, header = T)
  data$X <- NULL
  full_df <- rbind(full_df,data)
}

#### Data Processing
# Lag variables
full_df <- full_df %>% mutate(
  resp = ifelse(response == "c", 0, 1),
  prev_resp = lag(resp),
  sign_evi = ifelse(pre_dots_location_mean < 0, "left", "right"),
  prev_evidence = lag(pre_dots_location_mean),
  prev_sign_evi = ifelse(prev_evidence < 0, "left", "right"),
  prev_cj = lag(cj),
  prev_abs_evi = abs(prev_evidence),
  prev_condition = lag(condition),
  prev_acc = lag(accuracy),
)

# Data cleaning
full_df <- full_df %>% filter(response != "timed-out",
                              prev_resp != "timed-out",
                              trial != 1,
                              running != "practice")

# Missing data
sum(is.na(full_df)) 


### Exclusion
# abs evidence
absolute_evd <- abs(full_df$pre_dots_location_mean)
# Lapse rate < 95% (subset of trials with > 80 pixels evidence)
high_evidence <- subset(full_df, absolute_evd > 80)
lapse_rate <- high_evidence %>% group_by(sub) %>%
  summarise(lapse_rate = sum(accuracy == 1) / n())
exc_subs_lr <- lapse_rate %>% filter(lapse_rate < 0.95) %>% pull(sub)
# Confidence variability > 90% (same confidence response)
cj_variability <- full_df %>% group_by(sub) %>% 
  summarise(same_cj = max(table(cj)) / n()) 
exc_subs_cj <- cj_variability %>% filter(same_cj > 0.9) %>% pull(sub)
# Combining exclusion criteria
exc_subs <- union(exc_subs_lr, exc_subs_cj)
# Removing excluded subjects
full_df <- full_df %>% filter(!(sub %in% exc_subs))


### Scaling
# Maximal absolute evidence
max_abs_evidence <- full_df %>% group_by(sub) %>% 
  summarise(max_evidence = max(abs(pre_dots_location_mean))) %>% 
  summarise(global_max_evidence = max(max_evidence))
# Scaling evidence and previous evidence
full_df$scaled_evidence <- (full_df$pre_dots_location_mean) / max_abs_evidence$global_max_evidence
full_df$prev_scaled_evidence <- (full_df$prev_evidence) / max_abs_evidence$global_max_evidence
full_df$prev_scaled_abs_evi <- (full_df$prev_abs_evi) / max(full_df$prev_abs_evi)

### Creating 2 data frames
# DF with cj
df_cj <- full_df %>% filter(cj > -1)
df_cj <- df_cj %>% filter(prev_cj > -1)
# DF without cj
df_ncj <- full_df %>% filter(cj == -99)

# Scaling CJ
df_cj$scaled_cj <- (df_cj$cj - min(df_cj$cj)) / (max(df_cj$cj) - min(df_cj$cj))
df_cj$prev_scaled_cj <- (df_cj$prev_cj - min(df_cj$prev_cj)) / (max(df_cj$prev_cj) - min(df_cj$prev_cj))


# Adding variables
full_df <- full_df %>%
  group_by(sub, trial) %>%
  mutate(scaled_abs_evidence = abs(scaled_evidence),
         mean_acc = mean(accuracy, na.rm = TRUE),
         mean_rt = mean(rt, na.rm = TRUE))



#### Linear/Logistic Regression
### Model I
lm_cj <- lmer(data=df_cj, prev_scaled_cj ~
                prev_condition * prev_acc +
                (1|sub))
summary(lm_cj)
anova(lm_cj)


df_ncj$predicted_prev_cj <- predict(lm_cj, newdata=df_ncj)
df_ncj$predicted_prev_cj <- (df_ncj$predicted_prev_cj) / max(df_ncj$predicted_prev_cj)



### Model II
control_settings <- lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

model_cj <- glmer(resp ~ scaled_evidence +
                    prev_sign_evi * prev_scaled_abs_evi +
                    prev_cj +
                    (1 | sub),
                  data = df_cj,
                  family = binomial(link = "logit"),
                  control = control_settings)


summary(model_cj)
effects_cj <- ggpredict(model_cj, terms="scaled_evidence [all]")



### Model III
model_ncj <- glmer(resp ~ scaled_evidence +
                     prev_sign_evi * prev_scaled_abs_evi +
                     predicted_prev_cj + 
                     (1 | sub),
                   data = df_ncj,
                   family = binomial(link = "logit"),
                   control = control_settings)

summary(model_ncj)
effects_ncj <- ggpredict(model_ncj, terms="scaled_evidence [all]")


# VIF
vif(lm_cj)
vif(model_cj)
vif(model_ncj)

### Model Selection
# Model I
# Model without the random slope
lm_cj1 <- lmer(prev_scaled_cj ~
                 prev_condition * prev_acc +
                 (1|sub), data=df_cj)
# Model with the random slope
lm_cj2 <- lmer(prev_scaled_cj ~
                 prev_condition * prev_acc +
                 (1 + prev_condition + prev_acc| sub), data=df_cj)

anova(lm_cj1, lm_cj2)

# Model without the random slope for 'prev_condition'
lm_cj3 <- lmer(prev_scaled_cj ~
                 prev_condition * prev_acc +
                 (1 + prev_acc|sub), data=df_cj)

# Model without the random slope for 'prev_acc'
lm_cj4 <- lmer(prev_scaled_cj ~
                 prev_condition * prev_acc +
                 (1 + prev_condition|sub), data=df_cj)

# Compare the models
anova(lm_cj2, lm_cj3)
anova(lm_cj2, lm_cj4)

### We choose model lm_cj2 as the optimal model
summary(lm_cj2)
anova(lm_cj2)

# Model II
# Model without the random slopes
model_cj1 <- glmer(resp ~ scaled_evidence +
                     prev_sign_evi * prev_scaled_abs_evi +
                     prev_cj +
                     (1 | sub),
                   data = df_cj,
                   family = binomial(link = "logit"),
                   control = control_settings)

# Model with the random slopes
model_cj2 <- glmer(resp ~ scaled_evidence +
                     prev_sign_evi * prev_scaled_abs_evi +
                     prev_cj +
                     (1 + scaled_evidence + prev_sign_evi * prev_scaled_abs_evi + prev_cj | sub),
                   data = df_cj,
                   family = binomial(link = "logit"),
                   control = control_settings)

# Compare the models
anova(model_cj1, model_cj2)

model_cj3 <- glmer(resp ~ scaled_evidence +
                     prev_sign_evi * prev_scaled_abs_evi +
                     prev_cj +
                     (1 + scaled_evidence | sub),
                   data = df_cj,
                   family = binomial(link = "logit"),
                   control = control_settings)

model_cj4 <- glmer(resp ~ scaled_evidence +
                     prev_sign_evi * prev_scaled_abs_evi +
                     prev_cj +
                     (1 + prev_sign_evi * prev_scaled_abs_evi | sub),
                   data = df_cj,
                   family = binomial(link = "logit"),
                   control = control_settings)

model_cj5 <- glmer(resp ~ scaled_evidence +
                     prev_sign_evi * prev_scaled_abs_evi +
                     prev_cj +
                     (1 + prev_cj | sub),
                   data = df_cj,
                   family = binomial(link = "logit"),
                   control = control_settings)

anova(model_cj2, model_cj3)
anova(model_cj2, model_cj4)
anova(model_cj2, model_cj5)

### We choose model model_cj2 as the optimal model
summary(model_cj2)
anova(model_cj2)

# Model III
# Model without the random slopes
model_ncj1 <- glmer(resp ~ scaled_evidence +
                      prev_sign_evi * prev_scaled_abs_evi +
                      predicted_prev_cj +
                      (1 | sub),
                    data = df_ncj,
                    family = binomial(link = "logit"),
                    control = control_settings)

# Model with the random slopes
model_ncj2 <- glmer(resp ~ scaled_evidence +
                      prev_sign_evi * prev_scaled_abs_evi +
                      predicted_prev_cj +
                      (1 + scaled_evidence + prev_sign_evi * prev_scaled_abs_evi + predicted_prev_cj | sub),
                    data = df_ncj,
                    family = binomial(link = "logit"),
                    control = control_settings)

# Model with only the random slope for 'scaled_evidence'
model_ncj3 <- glmer(resp ~ scaled_evidence +
                      prev_sign_evi * prev_scaled_abs_evi +
                      predicted_prev_cj +
                      (1 + scaled_evidence | sub),
                    data = df_ncj,
                    family = binomial(link = "logit"),
                    control = control_settings)

# Model with only the random slope for 'prev_sign_evi' and 'prev_scaled_abs_evi'
model_ncj4 <- glmer(resp ~ scaled_evidence +
                      prev_sign_evi * prev_scaled_abs_evi +
                      predicted_prev_cj +
                      (1 + prev_sign_evi * prev_scaled_abs_evi | sub),
                    data = df_ncj,
                    family = binomial(link = "logit"),
                    control = control_settings)

# Model with only the random slope for 'predicted_prev_cj'
model_ncj5 <- glmer(resp ~ scaled_evidence +
                      prev_sign_evi * prev_scaled_abs_evi +
                      predicted_prev_cj +
                      (1 + predicted_prev_cj | sub),
                    data = df_ncj,
                    family = binomial(link = "logit"),
                    control = control_settings)

# Compare the models
anova(model_ncj1, model_ncj2)
anova(model_ncj2, model_ncj3)
anova(model_ncj2, model_ncj4)
anova(model_ncj2, model_ncj5)

### We choose model model_ncj2 as the optimal model
summary(model_ncj2)
anova(model_ncj2)


### Ploting the predicted effects
# Calculate predicted probabilities
effects_lm_cj <- ggpredict(lm_cj, terms = "prev_acc [all]")
plot1 <- plot(effects_lm_cj)
plot1 <- plot1 + labs(title = "A. Confidence Prediction Model",
                      x = "Accuracy",
                      y = "Predicted Confidence")

# Calculate predicted probabilities
df_ncj$predicted_prob <- predict(model_ncj, type = "response")
# Create a dataframe for the psychometric function plot
plot_data <- data.frame(mean_distance = df_ncj$scaled_evidence,
                        P_right = df_ncj$predicted_prob)

plot2 <- ggplot(plot_data, aes(x = mean_distance, y = P_right)) +
  geom_line(data = plot_data, alpha = 0.5) +
  labs(x = "Mean Distance", y = "P(right response)") +
  ggtitle("B. Model with Predicted Confidence") +
  theme_bw()

# Calculate predicted probabilities
df_cj$predicted_prob <- predict(model_cj, type = "response")
# Create a dataframe for the psychometric function plot
plot_data <- data.frame(mean_distance = df_cj$scaled_evidence,
                        P_right = df_cj$predicted_prob)


plot3 <- ggplot(plot_data, aes(x = mean_distance, y = P_right)) +
  geom_line(data = plot_data, alpha = 0.5) +
  labs(x = "Mean Distance", y = "P(right response)") +
  ggtitle("C. Model with Observed Confidence") +
  theme_bw()

grid.arrange(plot1, plot2, plot3, ncol=3)


#### Assumptions
# Defining a function for assumptions
check_assumptions_simple <- function(model, model_name) {
  
  # Normality
  ranef_resid <- ranef(model)[[1]]$`(Intercept)`
  hist(ranef_resid, main = paste("Histogram of Random Effects for", model_name),
       xlab = "Random Effects", breaks = 30)
  
  # Homoscedasticity
  fitted_ranef <- fitted(model)
  resid_ranef <- resid(model)
  
  plot(resid_ranef, fitted_ranef, main = paste("Residuals vs. Fitted values for", model_name),
       xlab = "Residuals", ylab = "Fitted values")
  abline(h = 0, col = "red", lty = 2)
  
}

# Assumptions of the model without confidence rating
check_assumptions_simple(model_ncj, "Model Without Confidence Rating")
# Assumptions of the model with confidence rating
check_assumptions_simple(model_cj, "Model With Confidence Rating")
# Assumptions of the lm_cj model
check_assumptions_simple(lm_cj, "lm_cj Model")




#### Descriptives
# Variables
colnames(full_df)


### Basic descriptives
describe(df_cj)


# Obtain unique values for each sub
ddf <- full_df %>%
  group_by(sub) %>%
  summarise(age = first(age),
            gender = first(gender),
            handedness = first(handedness))

# Age
describe(ddf$age)
# Gender
table(ddf$gender)
# Handedness
table(ddf$handedness)



#### Data Visualization
# Evidence bins
evidence_breaks <- quantile(full_df$scaled_evidence, seq(0, 1, 1 / 11))
full_df$evidence_bins <- cut(full_df$scaled_evidence,
                             breaks = evidence_breaks,
                             labels = (1:11),
                             include.lowest = TRUE)
# Calculate average distance for each evidence bin
mean_distance_per_bin <- full_df %>%
  group_by(evidence_bins) %>%
  summarise(mean_distance = mean(scaled_evidence, na.rm = TRUE))

evidence_breaks <- quantile(df_cj$scaled_evidence, seq(0, 1, 1 / 11))
df_cj$evidence_bins <- cut(df_cj$scaled_evidence,
                           breaks = evidence_breaks,
                           labels = (1:11),
                           include.lowest = TRUE)



# Creating data frames for plotting
df_rt <- full_df %>% group_by(sub, evidence_bins) %>% summarise(mean_rt = mean(rt))
df_acc <- full_df %>% group_by(sub, evidence_bins) %>% summarise(mean_acc = mean(accuracy))
df_cjp <- df_cj %>% group_by(sub, evidence_bins) %>% summarise(mean_cj = mean(scaled_cj))

# Color palette
color_palette <- colorRampPalette(rainbow(5))


# Summarise mean distance per bin
mean_distance_per_bin_summary <- mean_distance_per_bin %>%
  group_by(evidence_bins) %>%
  summarise(mean_distance = mean(mean_distance))

# Convert the bins to factor for proper ordering
mean_distance_per_bin_summary$evidence_bins <- as.factor(mean_distance_per_bin_summary$evidence_bins)

# Create the bar plot with reversed x-axis
mean_distance_plot <- ggplot(mean_distance_per_bin_summary,
                             aes(x=mean_distance, y=evidence_bins, fill=evidence_bins)) +
  geom_bar(stat="identity") +
  coord_flip() +
  scale_fill_brewer(palette="RdBu") +
  labs(y="Evidence Bins", x="Mean Distance",
       title="Average Mean Distance Per Bin") +
  theme_bw() +
  geom_point(data=mean_distance_per_bin_summary, aes(x=mean_distance, y=evidence_bins),
             shape=21, color="black", fill="black") +
  geom_line(data=mean_distance_per_bin_summary, aes(x=mean_distance, y=evidence_bins,
                                                    group=1), color="black")

mean_distance_plot



### Histogram
# ACC histogram
hist_acc <- ggplot(df_acc, aes(x=mean_acc)) + 
  geom_histogram(color="seagreen", fill="seagreen", alpha=0.5, bins=100) +
  labs(x="Accuracy", y="Frequency") +
  ggtitle("A. Accuracy") +
  theme_bw()

# CJ histogram
hist_cj <- ggplot(df_cjp, aes(x=mean_cj)) + 
  geom_histogram(color="seagreen", fill="seagreen", alpha=0.5, bins=100) +
  labs(x="Confidence", y="Frequency") +
  ggtitle("B. Confidence Judgment") +
  theme_bw()

# RT histogram
hist_rt <- ggplot(df_rt, aes(x=mean_rt)) + 
  geom_histogram(color="seagreen", fill="seagreen", alpha=0.5, bins=100) +
  labs(x="RT", y="Frequency") +
  ggtitle("C. Reaction Time") +
  theme_bw()

# Combining plots
grid.arrange(hist_acc, hist_cj, hist_rt, ncol=3)


### ACC, RT, and CJ plot
# ACC plot
plot_acc <- ggplot(df_acc, aes(x=evidence_bins, y=mean_acc, group=sub, color=sub)) + 
  geom_line(alpha=0.2) +
  stat_summary(fun=mean, geom="line", size=1, aes(group=1), color="black") +
  scale_color_gradientn(colours = color_palette(5), guide = 'none') +
  labs(x="Evidence bins", y="Accuracy", color="Subject") +
  ggtitle("A. Accuracy") +
  theme_bw()

# CJ plot
plot_cj <- ggplot(df_cjp, aes(x=evidence_bins, y=mean_cj, group=sub, color=sub)) + 
  geom_line(alpha=0.2) +
  stat_summary(fun=mean, geom="line", size=1, aes(group=1), color="black") +
  scale_color_gradientn(colours = color_palette(5), guide = 'none') +
  labs(x="Evidence bins", y="Confidence", color="Subject") +
  ggtitle("B. Confidence Judgment") +
  theme_bw()

# RT plot
plot_rt <- ggplot(df_rt, aes(x=evidence_bins, y=mean_rt, group=sub, color=sub)) + 
  geom_line(alpha=0.2) +
  stat_summary(fun=mean, geom="line", size=1, aes(group=1), color="black") +
  scale_color_gradientn(colours = color_palette(5), guide = 'none') +
  labs(x="Evidence bins", y="RT", color="Subject") +
  ggtitle("C. Reaction Time") +
  theme_bw()


# Combining plots
grid.arrange(plot_acc, plot_cj, plot_rt, ncol=3)


### CJ plot
# Calculate mean values per subject per confidence judgement
df_summary <- df_cj %>%
  group_by(sub, cj) %>%
  summarise(mean_acc = mean(accuracy, na.rm = TRUE),
            mean_rt = mean(rt, na.rm = TRUE),
            scaled_cj = mean(scaled_cj, na.rm = TRUE),
            mean_scaled_abs_evidence = mean(abs(scaled_evidence), na.rm = TRUE))

# CJ vs Absolute Evidence plot
plot_cj_abs_ev <- ggplot(df_summary, aes(x=scaled_cj, y=mean_scaled_abs_evidence, group=sub, color=sub)) + 
  geom_line(alpha=0.2) +
  stat_summary(fun=mean, geom="line", size=1, aes(group=1), color="black") +
  scale_color_gradientn(colours = color_palette(5), guide = 'none') +
  labs(x="Confidence", y="Average Absolute Scaled Evidence", color="Subject") +
  ggtitle("A. Absolute Evidence") +
  theme_bw()

# CJ vs ACC plot
plot_cj_acc <- ggplot(df_summary, aes(x=scaled_cj, y=mean_acc, group=sub, color=sub)) + 
  geom_line(alpha=0.2) +
  stat_summary(fun=mean, geom="line", size=1, aes(group=1), color="black") +
  scale_color_gradientn(colours = color_palette(5), guide = 'none') +
  labs(x="Confidence", y="Average Accuracy", color="Subject") +
  ggtitle("B. Accuracy") +
  theme_bw()

# CJ vs RT plot
plot_cj_rt <- ggplot(df_summary, aes(x=scaled_cj, y=mean_rt, group=sub, color=sub)) + 
  geom_line(alpha=0.2) +
  stat_summary(fun=mean, geom="line", size=1, aes(group=1), color="black") +
  scale_color_gradientn(colours = color_palette(5), guide = 'none') +
  labs(x="Confidence", y="Average RT", color="Subject") +
  ggtitle("C. Reaction Time") +
  theme_bw()

# Combining plots
grid.arrange(plot_cj_abs_ev, plot_cj_acc, plot_cj_rt, ncol=3)


# Add a new column 'sub' to df_cjp for individual participant data
df_cjp <- df_cj %>% group_by(sub, condition, evidence_bins) %>% 
  summarise(mean_cj = mean(scaled_cj, na.rm = TRUE))
df_cjp_acc <- df_cj %>% group_by(sub, accuracy, evidence_bins) %>% 
  summarise(mean_cj = mean(scaled_cj, na.rm = TRUE))
df_summary <- df_cjp %>% group_by(condition, evidence_bins) %>% 
  summarise(mean_cj = mean(mean_cj, na.rm = TRUE))
df_summary_acc <- df_cjp_acc %>% group_by(accuracy, evidence_bins) %>% 
  summarise(mean_cj = mean(mean_cj, na.rm = TRUE))

color_palette <- c("firebrick3", "cyan4")

# Plotting CJ vs Evidence Bins (by "weaker" and "stronger" conditions)
plot_condition <- ggplot() +
  # Add lines for each individual participant with alpha 0.2
  geom_line(data = df_cjp, aes(x = evidence_bins, y = mean_cj, group = interaction(sub, condition), color = condition), alpha = 0.2) +
  # Add lines for the mean values with different color
  geom_line(data = df_summary, aes(x = evidence_bins, y = mean_cj, group = condition, color = condition), size = 1) +
  geom_point(data = df_summary, aes(x = evidence_bins, y = mean_cj, color = condition)) +
  labs(x = "Evidence bins", y = "Average Confidence Rating", color = "Condition") +
  theme_bw() +
  ggtitle("A. Confidence, Grouped by Condition")


# Plotting CJ vs Evidence Bins (by accuracy)
plot_accuracy <- ggplot() +
  # Add lines for each individual participant with alpha 0.2
  geom_line(data = df_cjp_acc, aes(x = evidence_bins, y = mean_cj, group = interaction(sub, accuracy), color = as.factor(accuracy)), alpha = 0.2) +
  # Add lines for the mean values with different color
  geom_line(data = df_summary_acc, aes(x = evidence_bins, y = mean_cj, group = accuracy, color = as.factor(accuracy)), size = 1) +
  geom_point(data = df_summary_acc, aes(x = evidence_bins, y = mean_cj, color = as.factor(accuracy))) +
  labs(x = "Evidence bins", y = "Average Confidence Rating", color = "Accuracy") +
  scale_color_manual(values = color_palette, labels = c("Incorrect", "Correct")) +
  theme_bw() +
  ggtitle("B. Confidence, Grouped by Accuracy")

# Combine the two plots
grid.arrange(plot_condition, plot_accuracy, ncol=2)





# Add a new column 'sub' to df_cjp for individual participant data
df_cjp <- df_cj %>% group_by(sub, condition, evidence_bins, accuracy) %>% 
  summarise(mean_cj = mean(scaled_cj, na.rm = TRUE))

df_summary <- df_cjp %>% group_by(condition, evidence_bins, accuracy) %>% 
  summarise(mean_cj = mean(mean_cj, na.rm = TRUE))

# Filter data based on accuracy
df_cjp_correct <- df_cjp %>% filter(accuracy == 1)
df_cjp_incorrect <- df_cjp %>% filter(accuracy == 0)
df_summary_correct <- df_summary %>% filter(accuracy == 1)
df_summary_incorrect <- df_summary %>% filter(accuracy == 0)

# Plotting CJ vs Evidence Bins (for correct responses)
plot_correct <- ggplot() +
  # Add lines for each individual participant with alpha 0.2
  geom_line(data = df_cjp_correct, aes(x = evidence_bins, y = mean_cj, group = interaction(sub, condition), color = condition), alpha = 0.2) +
  # Add lines for the mean values with different color
  geom_line(data = df_summary_correct, aes(x = evidence_bins, y = mean_cj, group = condition, color = condition), size = 1) +
  geom_point(data = df_summary_correct, aes(x = evidence_bins, y = mean_cj, color = condition)) +
  labs(x = "Evidence bins", y = "Average Confidence Rating", color = "Condition") +
  theme_bw() +
  ggtitle("A. Correct Responses")

# Plotting CJ vs Evidence Bins (for incorrect responses)
plot_incorrect <- ggplot() +
  # Add lines for each individual participant with alpha 0.2
  geom_line(data = df_cjp_incorrect, aes(x = evidence_bins, y = mean_cj, group = interaction(sub, condition), color = condition), alpha = 0.2) +
  # Add lines for the mean values with different color
  geom_line(data = df_summary_incorrect, aes(x = evidence_bins, y = mean_cj, group = condition, color = condition), size = 1) +
  geom_point(data = df_summary_incorrect, aes(x = evidence_bins, y = mean_cj, color = condition)) +
  labs(x = "Evidence bins", y = "Average Confidence Rating", color = "Condition") +
  theme_bw() +
  ggtitle("B. Incorrect Responses")

# Combine the two plots
grid.arrange(plot_correct, plot_incorrect, ncol=2)



df_acc <- df_cj %>%
  group_by(sub, evidence_bins, condition) %>%
  summarise(mean_acc = mean(accuracy, na.rm = TRUE))

# Data preparation for reaction time
df_rt <- df_cj %>%
  group_by(sub, evidence_bins, condition) %>%
  summarise(mean_rt = mean(rt, na.rm = TRUE))

# Plotting Accuracy vs Evidence Bins
plot1 <- ggplot(df_acc, aes(x = evidence_bins, y = mean_acc, group = interaction(sub, condition), color = condition)) +
  geom_line(alpha = 0.2) +
  geom_line(data = df_acc %>% group_by(condition, evidence_bins) %>% summarise(mean_acc = mean(mean_acc, na.rm = TRUE)), aes(group = condition), size = 1) +
  labs(x = "Evidence Bins", y = "Average Accuracy", color = "Condition") +
  theme_bw() +
  ggtitle("A. Accuracy")

# Plotting Reaction Time vs Evidence Bins
plot2 <- ggplot(df_rt, aes(x = evidence_bins, y = mean_rt, group = interaction(sub, condition), color = condition)) +
  geom_line(alpha = 0.2) +
  geom_line(data = df_rt %>% group_by(condition, evidence_bins) %>% summarise(mean_rt = mean(mean_rt, na.rm = TRUE)), aes(group = condition), size = 1) +
  labs(x = "Evidence Bins", y = "Average Reaction Time", color = "Condition") +
  theme_bw() +
  ggtitle("B. Reaction Time")

# Combine the two plots
grid.arrange(plot1, plot2, ncol = 2)



# Data preparation
df_acc <- df_cj %>% 
  group_by(sub, condition) %>% 
  summarise(mean_accuracy = mean(accuracy, na.rm = TRUE),
            mean_prev_acc = mean(prev_acc, na.rm = TRUE),
            mean_prev_cj = mean(prev_scaled_cj, na.rm = TRUE), 
            mean_cj = mean(scaled_cj, na.rm = TRUE))
df_summary_acc <- df_acc %>% 
  group_by(condition) %>% 
  summarise(mean_accuracy = mean(mean_accuracy, na.rm = TRUE), mean_prev_acc = mean(mean_prev_acc, na.rm = TRUE))
df_summary_cj <- df_acc %>% 
  group_by(condition) %>% 
  summarise(mean_cj = mean(mean_cj, na.rm = TRUE), mean_prev_cj = mean(mean_prev_cj, na.rm = TRUE))


# Plot 1: mean prev_acc on Y and mean accuracy on X; grouped by condition
plot1 <- ggplot(df_acc, aes(x = mean_prev_acc, y = mean_accuracy, color = condition)) +
  geom_point(size = 3, alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, aes(group = condition, color = condition)) +
  labs(x = "Average Previous Accuracy", y = "Average Accuracy", color = "Condition") +
  theme_bw() +
  ggtitle("A. Sequential Effect of Accuracy")

# Plot 2: mean prev_acc on Y and mean scaled_cj on X; grouped by condition
plot2 <- ggplot(df_acc, aes(x = mean_prev_cj, y = mean_accuracy, color = condition)) +
  geom_point(size = 3, alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, aes(group = condition, color = condition)) +
  labs(x = "Average Previous Confidence Rating", y = "Average Accuracy",
       color = "Condition") + 
  theme_bw() +
  ggtitle("B. Sequential Effect of Confidence")

# Combine the two plots
grid.arrange(plot1, plot2, ncol=2)


