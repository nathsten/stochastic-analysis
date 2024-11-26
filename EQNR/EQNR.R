rm(list = ls())

library(jsonlite)
library(car)

eqnr = read.csv('./EQNR/EQNR.csv')
oil = read.csv('./EQNR/Oil.csv')


close_lag = lag(eqnr$Close_eqnr, n = 1)[-1]
close_true = eqnr$Close_eqnr[-length(eqnr$Close_eqnr)] 
time = lag(eqnr$Time_proxy, n = 1)[-1]
oil_true = oil$Close_oil[-length(oil$Close_oil)]
oil_lag = lag(oil$Close_oil, n = 1)[-1]

plot(c(0,650), c(250, 420))
lines(time, close_true)
regr1 = lm(close_true ~ time + close_lag + oil_lag)

summary(regr1)



pdiff_oil_lag = 100*(oil_lag[-1] - lag(oil_lag)[-length(oil_lag)+1]) / lag(oil_lag)[-length(oil_lag)+1]
time_lag = time[-1]
regr2 = lm(time_lag ~ pdiff_oil_lag)
summary(regr2)
plot(pdiff_oil_lag, time_lag)
abline(regr2)

diff = (close_true - close_lag)[-1]
regr3 = lm(diff ~ pdiff_oil_lag)
summary(regr3)
plot(pdiff_oil_lag, diff)
abline(regr3)

close_lag2 = close_lag[-length(close_lag)+1]
close_true2 = close_true[-length(close_true)+1]
regr4 = lm(close_true2 ~ time_lag + close_lag2 + pdiff_oil_lag)
summary(regr4)

## Skrive til JSON
lm_summary <- summary(regr4)

coefficients <- as.data.frame(lm_summary$coefficients[, c("Estimate", "Std. Error")])

ser <- lm_summary$sigma

summary_list <- list(
  coefficients = coefficients,
  ser = ser
)

summary_json <- toJSON(summary_list, pretty = TRUE)

write(summary_json, file = "./EQNR/EQNR_summary_Oil_drift.json")


cor(close_true, oil_true)
