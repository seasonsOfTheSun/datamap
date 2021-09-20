
library(tidyverse)

file = "data/processed/scale_59bbd09.csv"
df = read_csv(file)
p = ggplot(df, mapping = aes(x=scale, y=score,color=name)) + geom_point()
ggsave("figs/scale_score.png", p)
ggsave("figs/scale_score.svg", p)
p = p + xlab("Cluster radius")

p = ggplot(df, mapping = aes(x=scale, y=time,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Time (s)")
p = p + xlab("Cluster radius")
ggsave("figs/scale_time.png", p)
ggsave("figs/scale_time.svg", p)

p = ggplot(df, mapping = aes(x=scale, y=max_memory,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Memory (bits)")
p = p + xlab("Cluster radius")
ggsave("figs/scale_memory.png", p)
ggsave("figs/scale_memory.png", p)



# size
file = "data/processed/size_59bbd09.csv"
df = read_csv(file)
p = ggplot(df, mapping = aes(x=size, y=score,color=name)) + geom_point()
ggsave("figs/size_score.png", p)
ggsave("figs/size_score.svg", p)
p = p + xlab("Cluster size")

p = ggplot(df, mapping = aes(x=size, y=time,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Time (s)")
p = p + xlab("Cluster size")
ggsave("figs/size_time.png", p)
ggsave("figs/size_time.svg", p)

p = ggplot(df, mapping = aes(x=size, y=max_memory,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Memory (bits)")
p = p + xlab("Cluster size")
ggsave("figs/size_memory.png", p)
ggsave("figs/size_memory.svg", p)

# dimension
file = "data/processed/dimension_5d490a7.csv"
df = read_csv(file)
p = ggplot(df, mapping = aes(x=dimension, y=score,color=name)) + geom_point()
p = p + xlab("Original Dimensions")
ggsave("figs/dimension_score.png", p)
ggsave("figs/dimension_score.svg", p)

p = ggplot(df, mapping = aes(x=dimension, y=time,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Time (s)")
p = p + xlab("Original Dimensions")
ggsave("figs/dimension_time.png", p)
ggsave("figs/dimension_time.svg", p)

p = ggplot(df, mapping = aes(x=dimension, y=max_memory,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Memory (bits)")
p = p + xlab("Original Dimensions")
ggsave("figs/dimension_memory.png", p)
ggsave("figs/dimension_memory.svg", p)

# final dimension
file = "data/processed/final_dimension_5d490a7.csv"
df = read_csv(file)
p = ggplot(df, mapping = aes(x=final_dimension, y=score,color=name)) + geom_point()
p = p + xlab("n. Features")
ggsave("figs/dimension_score.png", p)
ggsave("figs/dimension_score.svg", p)

p = ggplot(df, mapping = aes(x=final_dimension, y=time,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Time (s)")
p = p + xlab("n. Features")
ggsave("figs/dimension_time.png", p)
ggsave("figs/dimension_time.svg", p)

p = ggplot(df, mapping = aes(x=final_dimension, y=max_memory,color=name)) + geom_point()
p = p + scale_y_log10() + ylab("Memory (bits)")
p = p + xlab("n. Features")
ggsave("figs/dimension_memory.png", p)
ggsave("figs/dimension_memory.svg", p)

