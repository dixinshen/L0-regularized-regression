library(MASS)
library(glmnet)
source("/Users/dixinshen/Dropbox/Proximal/L0_proximal.R")

# independent featrues
set.seed(201911)

N <- 100
p <- 200
b0 <- 0.1
d <- 10
w <- c(rep(0.5,d), rep(0.01,p-d))
beta <- sapply(w, function(x) rnorm(1,0,x))
beta[(d+1):p] <- 0

mean_x <- rep(0, p)
corr_x <- 0.1
cov_x <- matrix(NA, nrow = p, ncol = p)
for (j in 1:p) {
    for (i in 1:p) {
        cov_x[i, j] <- corr_x^abs(i - j)
    }
}
x <- mvrnorm(n = N, mu = mean_x, Sigma = cov_x)
xval <- mvrnorm(n = 1000, mu = mean_x, Sigma = cov_x)
snr <- 1
signal_x <- drop(crossprod(beta, cov_x) %*% beta)
var_y <- signal_x / snr
y <- b0 + drop(x %*% beta) + sqrt(var_y) * rnorm(n = N)
yval <- b0 + drop(xval %*% beta) + sqrt(var_y) * rnorm(n = 1000)

res <- prox_L0(x, y, d=d)
betas <- res$beta
res$iter
1 - sum((yval - res$intercept - drop(xval%*%betas))^2)/sum((yval-mean(yval))^2)

res_w <- prox_L0(x, y, d=d, W=1/w)
betas_w <- res_w$beta
res_w$iter
1 - sum((yval - res_w$intercept - drop(xval%*%betas_w))^2)/sum((yval-mean(yval))^2)

rbind(beta, betas, betas_w)


set.seed(201911)
N <- 100
p <- 200
b0 <- 0.1
d <- 10
tpr <- numeric()
fpr <- numeric()
tpr_w <- numeric()
fpr_w <- numeric()
tpr_lasso <- numeric()
fpr_lasso <- numeric()
r2 <- numeric()
r2_w <- numeric()
r2_lasso <- numeric()
nselect <- rep(0, d)
nselect_w <- rep(0, d)
nselect_lasso <- rep(0, d)

mean_x <- rep(0, p)
corr_x <- 0.1
cov_x <- matrix(NA, nrow = p, ncol = p)
for (j in 1:p) {
    for (i in 1:p) {
        cov_x[i, j] <- corr_x^abs(i - j)
    }
}

for (i in 1:100) {
    w <- c(rep(0.5,d), rep(0.01,p-d))
    beta <- sapply(w, function(x) rnorm(1,0,x))
    beta[(d+1):p] <- 0
    
    x <- mvrnorm(n = N, mu = mean_x, Sigma = cov_x)
    xval <- mvrnorm(n = 1000, mu = mean_x, Sigma = cov_x)
    snr <- 1
    signal_x <- drop(crossprod(beta, cov_x) %*% beta)
    var_y <- signal_x / snr
    y <- b0 + drop(x %*% beta) + sqrt(var_y) * rnorm(n = N)
    yval <- b0 + drop(xval %*% beta) + sqrt(var_y) * rnorm(n = 1000)
    
    # unweighted
    res <- prox_L0(x, y, d = d, k_rho = 20)
    betas <- res$beta
    tpr[i] <- sum(betas!=0 & beta!=0)/d
    fpr[i] <- sum(betas!=0 & beta==0)/(p-d)
    nselect <- nselect + (betas[1:d]!=0)
    r2[i] <- 1 - sum((yval - res$intercept - drop(xval%*%betas))^2)/sum((yval-mean(yval))^2)
    
    # weighted, d=truth
    res_w <- prox_L0(x, y, d=d, W=1/w, rho_max = 1e6)
    betas_w <- res_w$beta
    tpr_w[i] <- sum(betas_w!=0 & beta!=0)/d
    fpr_w[i] <- sum(betas_w!=0 & beta==0)/(p-d)
    nselect_w <- nselect_w + (betas_w[1:d]!=0)
    r2_w[i] <- 1 - sum((yval - res_w$intercept - drop(xval%*%betas_w))^2)/sum((yval-mean(yval))^2)
    
    # L1
    res1 <- glmnet(x=x, y=y)
    betas_lasso <- as.matrix(res1$beta)
    row.names(betas_lasso) <- NULL
    pred_lasso <- as.matrix(predict(res1, newx = xval, type = "link"))
    r21 <- drop( apply(pred_lasso, 2, function(a,y) 1-sum((a-yval)^2)/sum((yval-mean(yval))^2)) )
    lamInd_opt <- which.max(r21)
    r2_lasso[i] <- r21[lamInd_opt]
    betas_lasso <- betas_lasso[,lamInd_opt]
    tpr_lasso[i] <- sum(betas_lasso!=0 & beta!=0)/d
    fpr_lasso[i] <- sum(betas_lasso!=0 & beta==0)/(p-d)
    nselect_lasso <- nselect_lasso + (betas_lasso[1:d]!=0)
}

mean(r2)
mean(r2_w)
mean(r2_lasso)
mean(tpr)
mean(fpr)
mean(tpr_w)
mean(fpr_w)
mean(tpr_lasso)
mean(fpr_lasso)
nselect
nselect_w
nselect_lasso


# correlated beta estimates
set.seed(201911)
N <- 100
p <- 200
b0 <- 0.1
d <- 10
variance <- c(rep(0.5,d), rep(0.01,p-d))
corr_b <- 0.6
cov_b <- matrix(NA, nrow = p, ncol = p)
for (j in 1:p) {
    for (i in 1:p) {
        cov_b[i, j] <- corr_b^abs(i - j)
    }
}
diag(cov_b) <- variance
cov_b <- nearPD(cov_b, keepDiag = T)
cov_b <- as.matrix(cov_b$mat)
beta <- mvrnorm(1, mu=rep(0,p), Sigma=cov_b)
beta[(d+1):p] <- 0
W <- solve(cov_b)

mean_x <- rep(0, p)
corr_x <- 0.1
cov_x <- matrix(NA, nrow = p, ncol = p)
for (j in 1:p) {
    for (i in 1:p) {
        cov_x[i, j] <- corr_x^abs(i - j)
    }
}
x <- mvrnorm(n = N, mu = mean_x, Sigma = cov_x)
xval <- mvrnorm(n = 1000, mu = mean_x, Sigma = cov_x)
snr <- 1
signal_x <- drop(crossprod(beta, cov_x) %*% beta)
var_y <- signal_x / snr
y <- b0 + drop(x %*% beta) + sqrt(var_y) * rnorm(n = N)
yval <- b0 + drop(xval %*% beta) + sqrt(var_y) * rnorm(n = 1000)

res <- prox_L0(x, y, d=d, k_rho = 20)
betas <- res$beta
res$iter
1 - sum((yval - res$intercept - drop(xval%*%betas))^2)/sum((yval-mean(yval))^2)

res_w <- prox_L0(x, y, d=d, W=W, rho_max = 1e6)
betas_w <- res_w$beta
res_w$iter
1 - sum((yval - res_w$intercept - drop(xval%*%betas_w))^2)/sum((yval-mean(yval))^2)

res_dw <- prox_L0(x, y, d=d, W=1/variance)
betas_dw <- res_dw$beta
res_dw$iter
1 - sum((yval - res_dw$intercept - drop(xval%*%betas_dw))^2)/sum((yval-mean(yval))^2)

rbind(beta, betas, betas_dw, betas_w)




