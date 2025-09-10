library(tidyverse)


plot_data <- function(filename) {
  df <- read_csv(filename)

  ggplot(df,
  aes(
    x=fct_reorder(.f=word, .x=extracted_probability, .na_rm=TRUE),
    y=extracted_probability
  )) +
    geom_boxplot() +
    coord_flip() +
    labs(
      x=NULL,
      y=NULL,
      title="What probability do AI models give words?"
    ) +
    theme_bw(base_size = 16)
}

same_plot <- function(filename) {
  df <- read_csv(filename)

  data <- df |>
    rename(
      prob=extracted_probability
    ) |>
    group_by(
      word
    ) |>
    summarize(
      n=n(),
      mean=mean(prob),
      sd=sd(prob),
      low = mean-sd,
      high = mean + sd,
      .groups="drop"
    ) |>
    arrange(mean)

    ggplot(data, aes(
      x=fct_reorder(.f=word, .x=mean, .na_rm=TRUE),
      y=mean
    )) +
      geom_point(shape = 23, size = 3, fill = "white", stroke = 0.8) +
      geom_point(aes(y=low)) +
      geom_point(aes(y=high)) +
      geom_segment(aes(y = low, yend = high, xend = word), linetype = "dashed", linewidth = 0.6) +
      coord_flip() +
      labs(
        x=NULL,
        y=NULL,
        title="How do AI models use estimative probabilities?"
      ) +
      theme_bw(base_size=16)

}


plot_data("EstimativeWordsModels.csv")
plot_data("EstimativeWordsModels2.csv")


same_plot("EstimativeWordsModels2.csv")

df |>
  filter(
    word=="Impossible"
  ) |>
  arrange(
    -extracted_probability
  ) |>
  select(
    model, word, extracted_probability
  )
