rm(list = ls())

data = read.csv('./AKER/AKER_Hedge.csv')

t = data$X + 1
n = length(t)

Y_t = data$AKER
Y_lag = c(NA, Y_t[1:(n-1)])

Y_hat_aker = lm(Y_t ~ t + Y_lag)
summary(Y_hat_aker)

Y_t = data$FRO
Y_lag = c(NA, Y_t[1:(n-1)])

Y_hat_fro = lm(Y_t ~ Y_lag)
summary(Y_hat_fro)

mean(Y_t[2:n]/Y_lag[2:n])

drift_fro = Y_t + 0.0009225*t - 0.9911782*Y_lag

plot(t,drift_fro)
regr_drift_fro = lm(drift_fro ~ t)
summary(regr_drift_fro)

## Skrive til JSON
lm_summary <- summary(Y_hat_fro)

coefficients <- as.data.frame(lm_summary$coefficients[, c("Estimate", "Std. Error")])

ser <- lm_summary$sigma

summary_list <- list(
  coefficients = coefficients,
  ser = ser
)

summary_json <- toJSON(summary_list, pretty = TRUE)

write(summary_json, file = "./AKER/FRO_est.json")

qnorm(0.75)
