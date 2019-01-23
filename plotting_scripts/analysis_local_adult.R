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


df_bbox <- read.csv("data/bbox_unfairness_adult.csv", header=T)
df      <- read.csv("data/badml_unfairness_adult.csv", header=T)




df_1 <- df %>% filter(fidelity_0.1 == "True")
badml_1 <- df_1$unfairness_0.1 

df_3 <- df %>% filter(fidelity_0.3 == "True")
badml_3 <- df_3$unfairness_0.3 


df_5 <- df %>% filter(fidelity_0.5 == "True")
badml_5 <- df_5$unfairness_0.5 


df_7 <- df %>% filter(fidelity_0.7 == "True")
badml_7 <- df_7$unfairness_0.7 


df_9 <- df %>% filter(fidelity_0.9 == "True")
badml_9 <- df_9$unfairness_0.9 

bbox <- df_bbox$unfairness



df_merge <- data.frame(x = c(bbox, badml_1, badml_3, badml_5, badml_7, badml_9), ggg=factor(rep(1:6, c(nrow(df_bbox), nrow(df_1), nrow(df_3), nrow(df_5), nrow(df_7), nrow(df_9)))))
ggplot(df_merge, aes(x, colour = ggg,linetype=ggg)) + 
  stat_ecdf(geom = "step") + 
  theme_bw(base_size=13) +
  scale_color_hue(name="", labels=c('black-box',expression(paste(beta,"=0.1")), expression(paste(beta,"=0.3")), expression(paste(beta,"=0.5")), expression(paste(beta,"=0.7")), expression(paste(beta,"=0.9")) )) + labs(x = "Unfairness", y = "Proportion of users") +
  scale_linetype(name="", labels=c('black-box',expression(paste(beta,"=0.1")), expression(paste(beta,"=0.3")), expression(paste(beta,"=0.5")), expression(paste(beta,"=0.7")), expression(paste(beta,"=0.9")) )) + labs(x = "Unfairness", y = "Proportion of users") 
ggsave("graphs/local_adult.pdf", dpi=300, width=4.5, height=3)

print(nrow(df))

print(100*nrow(df_1)/nrow(df))
print(100*nrow(df_3)/nrow(df))
print(100*nrow(df_5)/nrow(df))
print(100*nrow(df_7)/nrow(df))
print(100*nrow(df_9)/nrow(df))

