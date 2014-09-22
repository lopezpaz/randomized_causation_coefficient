library(mixtools)

write('SampleID,A,B',file='syn_pairs.csv')

write_pair <- function(x,y) {
  write(paste(i, paste(x,collapse=" "), paste(y,collapse=" "), sep=","),
  append=TRUE, file="syn_pairs.csv")
}

cause <- function(n,k=5,p1=5,p2=5) {
  w <- abs(runif(k))
  w <- w/sum(w)
  m <- rnorm(k,0,p1)
  s <- abs(rnorm(k,1,p2))
  scale(rnormmix(n,w,m,s))
}

noise <- function(n,v) {
  v*rnorm(n)
}

mechanism <- function(x,d=10) {
  g <- seq(min(x)-sd(x),max(x)+sd(x),length.out=d)
  function(z) predict(smooth.spline(g,rnorm(d)),z)$y
}

N  <- 5000

set.seed(0)

for(i in 1:N) {
  x <- cause(1000)
  f <- mechanism(x)
  e <- noise(length(x),runif(1))
  write_pair(x,scale(f(x))+e)
}

write.table(cbind(1:(1*N),1,0),sep=",", quote=FALSE,
col.names=F,row.names=F,file="syn_target.csv")
