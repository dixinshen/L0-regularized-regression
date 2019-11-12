library(MASS)
library(glmnet)
source("/Users/dixinshen/Dropbox/Proximal/L0_proximal.R")

# simulate data
set.seed(201911)

N <- 100
p <- 200
b0 <- 0.1 # intercept
beta <- rep(0, p)
# the first 10 betas are non-zero
beta[1:10] <- rep(0.5, 10)

mean_x <- rep(0, p)
corr_x <- 0.1
cov_x <- matrix(NA, nrow = p, ncol = p)
for (j in 1:p) {
    for (i in 1:p) {
        cov_x[i, j] <- corr_x^abs(i - j)
    }
}
x <- mvrnorm(n = N, mu = mean_x, Sigma = cov_x)
xval <- mvrnorm(n = 1000, mu = mean_x, Sigma = cov_x) # validation set

snr <- 1
signal_x <- drop(crossprod(beta, diag(p)) %*% beta)
var_y <- signal_x / snr
y <- b0 + drop(x %*% beta) + sqrt(var_y) * rnorm(n = N)
yval <- b0 + drop(xval %*% beta) + sqrt(var_y) * rnorm(n = 1000)

res <- prox_L0(x, y, d = 10, accelerate = T)
res$intercept
res$beta
res$iter
1 - sum((y - res$intercept - drop(x%*%res$beta))^2)/sum((y-mean(y))^2)
1 - sum((yval - res$intercept - drop(xval%*%res$beta))^2)/sum((yval-mean(yval))^2)


set.seed(201911)
r2sim <- numeric()
r2sim_glmnet <- numeric()
for (j in 1:100) {
    x <- mvrnorm(n = N, mu = mean_x, Sigma = cov_x)
    xval <- mvrnorm(n = 1000, mu = mean_x, Sigma = cov_x) # validation set
    
    snr <- 1
    signal_x <- drop(crossprod(beta, diag(p)) %*% beta)
    var_y <- signal_x / snr
    y <- b0 + drop(x %*% beta) + sqrt(var_y) * rnorm(n = N)
    yval <- b0 + drop(xval %*% beta) + sqrt(var_y) * rnorm(n = 1000)
    
    
    # cross validation to compare L0 to L1
    r2 <- numeric()
    intercept <- numeric()
    betas <- mat.or.vec(20, p)
    for (d in 1:20) {
        fit <- prox_L0(x, y, d = d)
        intercept[d] <- fit$intercept
        betas[d,] <- fit$beta
        pred <- intercept[d] + drop(xval %*% betas[d, ])
        r2[d] <- 1 - sum((yval - pred)^2)/sum((yval-mean(yval))^2)
    }
    
    d_opt <- which.max(r2); d_opt
    r2[d_opt]
    beta_opt <- betas[d_opt, ]; beta_opt
    
    r2sim[j] <- r2[d_opt]
    
    # L1
    res1 <- glmnet(x=x, y=y)
    betas_glmnet <- as.matrix(res1$beta)
    row.names(betas_glmnet) <- NULL
    
    pred_glmnet <- as.matrix(predict(res1, newx = xval, type = "link"))
    r2_glmnet <- drop( apply(pred_glmnet, 2, function(a,y) 1-sum((a-yval)^2)/sum((yval-mean(yval))^2)) )
    lamInd_opt <- which.max(r2_glmnet); lamInd_opt
    r2_glmnet[lamInd_opt]
    betas_glmnet[,lamInd_opt]
    
    r2sim_glmnet[j] <- r2_glmnet[lamInd_opt]
}

mean(r2sim)
mean(r2sim_glmnet)


# compare to best subset selection
set.seed(201905)

N <- 200
p <- 12
b0 <- 0.1 # intercept
beta <- rep(0, p)
# the first 4 betas are non-zero
beta[1:4] <- c(1, 0.8, 0.6, 0.4)

mean_x <- rep(0, p)
corr_x <- 0.1
cov_x <- matrix(NA, nrow = p, ncol = p)
for (j in 1:p) {
    for (i in 1:p) {
        cov_x[i, j] <- corr_x^abs(i - j)
    }
}
x <- mvrnorm(n = N, mu = mean_x, Sigma = cov_x)
xval <- mvrnorm(n = 1000, mu = mean_x, Sigma = cov_x) # validation set

snr <- 1
signal_x <- drop(crossprod(beta, cov_x) %*% beta)
var_y <- signal_x / snr
y <- b0 + drop(x %*% beta) + sqrt(var_y) * rnorm(n = N)
yval <- b0 + drop(xval %*% beta) + sqrt(var_y) * rnorm(n = 1000)

res_bst <- prox_L0(x, y, d = 4)
res_bst$intercept
res_bst$beta
res_bst$iter

# validation L0
r2_bst <- numeric()
intercept_bst <- numeric()
betas_bst <- mat.or.vec(12, p)
for (d in 1:12) {
    fit <- prox_L0(x, y, d = d)
    intercept_bst[d] <- fit$intercept
    betas_bst[d,] <- fit$beta
    pred <- intercept_bst[d] + drop(xval %*% betas_bst[d, ])
    r2_bst[d] <- 1 - sum((yval - pred)^2)/sum((yval-mean(yval))^2)
}

d_opt_bst <- which.max(r2_bst); d_opt_bst
r2_bst[d_opt_bst]
inter_opt_bst <- intercept_bst[d_opt_bst]; inter_opt_bst
beta_opt_bst <- betas_bst[d_opt_bst, ]; beta_opt_bst

library(mlr)
dat <- data.frame(y=y, x=x)
lm_subset_tsk = makeRegrTask(id = "best subset", data = dat, target = "y")
learner_lm = makeLearner("regr.lm", fix.factors.prediction = TRUE)
rdesc = makeResampleDesc("CV", iters=10L)
ctrl_best = makeFeatSelControlExhaustive(max.features = 12) 
lm_subset = selectFeatures(learner = learner_lm, 
                          task = lm_subset_tsk, resampling = rdesc, 
                          measures = rsq, control = ctrl_best, 
                          show.info = FALSE)
lm_subset
dat <- dat[ , c(1:5)]
fitlm <- lm(y ~., data=dat)
fitlm

