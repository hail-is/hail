package org.broadinstitute.hail.io.annotators

import org.apache.hadoop
import org.broadinstitute.hail.Utils._

object SampleListAnnotator {
  def apply(filename: String, hConf: hadoop.conf.Configuration): Map[String, Boolean] = {
    readLines(filename, hConf) { lines =>
      if (lines.isEmpty)
        fatal("Empty file")

      val m = lines.map {
        _.transform { line =>
          (line.value, true)
        }
      }.toMap
      m
    }
  }
}
