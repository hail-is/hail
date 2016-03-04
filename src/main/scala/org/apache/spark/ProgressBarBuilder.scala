package org.apache.spark

import org.apache.spark.ui.ConsoleProgressBar

object ProgressBarBuilder {
  def build(sc: SparkContext): ConsoleProgressBar =
    new ConsoleProgressBar(sc)
}
