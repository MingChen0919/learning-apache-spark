set.seed(123)
X = data.frame(rnorm(20, 3, 5), rnorm(20, 4, 10))

U = svd(X)$u
S = diag(svd(X)$d)
V = t(svd(X)$v)


u11 = U[1,1]; u12 = U[1,2]
s11 = S[1,1]; s21 = S[2,1]
v11 = V[1,1]; v21 = V[2,1]
us11 = (U %*% S)[1,1]; us12 = (U %*% S)[1,2]
usv11 = (U %*% S %*% V)[1,1]; usv12 = (U %*% S %*% V)[1,2]

# plot(X[,1], X[,2], xlim = c(-8, 18), ylim = c(-8,18))
plot(X[,1], X[,2], xlim=c(-20, 40), ylim=c(-20, 40))
x1 = c(X[1,1], X[1,2])
points(x1[1], x1[2], col="red", pch=16)
arrows(0, 0, x1[1], x1[2], length=0.05)


points(u11, u12, pch=15, col="blue")
arrows(0,0, u11, u12, length=0.05)
points(s11, s21, pch=15, col="blue")
points(us11, us12, pch=15, col="green")
arrows(0,0, us11, us12, length=0.05)

points(us11, us12, pch=15, col="green")
points(v11, v21, pch=15, col="green")
points(usv11, usv12, pch=15, col="orange")
arrows(0,0, usv11, usv12, length=0.05)

