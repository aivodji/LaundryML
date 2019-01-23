#options(warn=1)


library(readr)
library(dplyr)
library(ggplot2)
library(scales)
library(grid)
library(RColorBrewer)
library(digest)
library(readr)
library(stringr)


df <- read.csv("data/both.csv", header=T)

df$which <- factor(df$which, labels = c("Adult~Income", "ProPublica~Recidivism"))

df$lambda <- factor(df$lambda, labels = c(expression(paste(lambda,"=0.005")), expression(paste(lambda,"=0.01"))))

g <- ggplot(df) +
        geom_point(aes(unfairness_train, fidelity_train, shape = as.factor(beta), col = as.factor(beta))) + 
        theme_bw(base_size=13) +
        labs(x="Unfairness", y="Fidelity", color  = expression(beta), shape  = expression(beta)) + 
        scale_x_continuous(breaks=seq(0,0.2,by=0.1),limits=c(0,0.2))

g + facet_wrap(which ~ lambda,nrow=1,labeller = label_parsed)

ggsave("graphs/both_train.png", dpi=300, width=9, height=3)


g <- ggplot(df) +
        geom_point(aes(unfairness_test, fidelity_test, shape = as.factor(beta), col = as.factor(beta))) + 
        theme_bw(base_size=13) +
        labs(x="Unfairness", y="Fidelity", color  = expression(beta), shape  = expression(beta)) + 
        scale_x_continuous(breaks=seq(0,0.2,by=0.1),limits=c(0,0.2))

g + facet_wrap(which ~ lambda,nrow=1,labeller = label_parsed)

ggsave("graphs/both_test.png", dpi=300, width=9, height=3)




