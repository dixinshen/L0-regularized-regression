# L0 constrained linear regression using proximal distance algorithm

prox_L0 <- function(x, y, d, rho_0 = 1, rho_inc = 1.2, rho_max = 1000, k_rho = 100, W = 1,
                    iter_max = 10000, tolerance = 1e-10, standardize = TRUE, accelerate = FALSE) {
    N <- NROW(y)
    p <- NCOL(x)
    v <- rep(1/N, N)
    ym <- mean(y)
    y_cent <- y - ym
    
    xm <- colMeans(x)
    x <- sweep(x, 2, xm, "-")
    if (standardize == TRUE) { 
        xs <- drop(sqrt(crossprod(v, x^2)))
        x <- sweep(x, 2, xs, "/")
    } else {
        xs <- rep(1, p)
    }
    
    if (is.vector(W) && length(W) == 1) {
        W <- rep(1, p)
    } 
    
    # initialize
    rho <- rho_0
    dist0 <- 1e35
    
    if (accelerate == FALSE) {
        b_prime <- rep(0, p)
        dist_prime <- dist0
        
        for (i in 1:iter_max) {
            b_proj <- b_prime
            b_proj[order(abs(b_prime), decreasing = T)[(d+1):p]] <- 0
            
            if (is.vector(W)) {
                b_current <- drop( solve( (t(x)%*%x + rho*W*diag(p)), (t(x)%*%y_cent + rho*W*b_proj) ) ) 
                dist_current <- sqrt( W*(b_current - b_proj)^2 ) 
            } else if (is.matrix(W)) {
                b_current <- drop( solve( (t(x)%*%x + rho*W), (t(x)%*%y_cent + rho*W%*%b_proj) ) ) 
                dist_current <- sqrt( drop(crossprod(b_current-b_proj, W) %*% (b_current-b_proj)) )
            }
            
            dlb <- max(abs(b_current - b_prime))
            dldist <- abs(dist_current - dist_prime)
            
            if (dlb < tolerance && dldist < tolerance) {
                break
            } else {
                dist_prime <- dist_current
                b_prime <- b_current
                
                if (i %% k_rho == 0) {
                    rho <- min(rho_max, rho*rho_inc)
                }
            }
        }
        
        beta <- b_current
        beta[order(abs(b_current), decreasing = T)[(d+1):p]] <- 0
        beta <- beta / xs
        b0 <- ym - drop(crossprod(xm, beta))
        
        if (dlb < tolerance && dldist < tolerance) {
            return ( list(intercept = b0, beta = beta, iter = i, rho = rho) )
        } else {
            warning("Max iteration reached, not converged. \n")
            return ( list(intercept = b0, beta = beta, iter = i, rho = rho) ) 
        }
        
    } else if (accelerate == TRUE) {
        b_minus_2 <- rep(0, p)
        b_minus_1 <- rep(0, p)
        dist_prime <- dist0
        
        for (i in 2:iter_max) {
            z_current <- b_minus_1 + ((i-1)/(i+2)) * (b_minus_1 - b_minus_2)
            b_minus_2 <- b_minus_1
            z_proj <- z_current
            z_proj[order(abs(z_current), decreasing = T)[(d+1):p]] <- 0
            
            if (is.vector(W)) {
                b_current <- drop( solve( (t(x)%*%x + rho*W*diag(p)), (t(x)%*%y_cent + rho*W*z_proj) ) )
                dist_current <- sqrt( sum(W*(b_current - z_proj)^2) )
            } else if (is.matrix(W)) {
                b_current <- drop( solve( (t(x)%*%x + rho*W), (t(x)%*%y_cent + rho*W%*%z_proj) ) )
                dist_current <- sqrt( drop(crossprod(b_current-z_proj, W) %*% (b_current-z_proj)) )
            }
            
            dlb <- max(abs(b_current - b_minus_1))
            dldist <- abs(dist_current - dist_prime)
            
            if (dlb < tolerance & dldist < tolerance) {
                break
            } else {
                dist_prime <- dist_current
                b_minus_1 <- b_current
                
                if (i %% k_rho == 0) {
                    rho <- min(rho_max, rho*rho_inc)
                }
            }
        }
        
        beta <- b_current
        beta[order(abs(b_current), decreasing = T)[(d+1):p]] <- 0
        beta <- beta / xs
        b0 <- ym - drop(crossprod(xm, beta))
        
        if (i == iter_max) {
            warning("Max iteration reached, not converged. \n")
            return ( list(intercept = b0, beta = beta, iter = i, rho = rho) )
        } else {
            return ( list(intercept = b0, beta = beta, iter = i, rho = rho) ) 
        }
    }
}


