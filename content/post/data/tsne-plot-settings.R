setMethod("plot","VectorSpaceModel",function(x,method="tsne",...) {
if (method=="tsne") {
message("Attempting to use T-SNE to plot the vector representation")
message("Cancel if this is taking too long")
message("Or run 'install.packages' tsne if you don't have it.")
x = as.matrix(x)
short = x[1:min(100,nrow(x)),]
m = tsne::tsne(short,...)
graphics::plot(m,type='n',main="A two dimensional reduction of the vector space model using t-SNE")
graphics::text(m,rownames(short),cex = ((400:1)/200)^(1/3))
rownames(m)=rownames(short)
silent = m
} else if (method=="pca") {
vectors = stats::predict(stats::prcomp(x))[,1:2]
graphics::plot(vectors,type='n')
graphics::text(vectors,labels=rownames(vectors))
}
})
