package org.broadinstitute.hail.utils

case class Line(value: String, position: Option[Int], filename: String) {
  def transform[T](f: Line => T): T = {
    try {
      f(this)
    } catch {
      case e: Exception =>
        val lineToPrint =
          if (value.length > 62)
            value.take(59) + "..."
          else
            value
        val msg = if (e.isInstanceOf[FatalException])
          e.getMessage
        else
          s"caught $e"
        log.error(
          s"""
             |$filename${position.map(ln => ":" + (ln + 1)).getOrElse("")}: $msg
             |  offending line: $value""".stripMargin)
        fatal(
          s"""
             |$filename${position.map(ln => ":" + (ln + 1)).getOrElse("")}: $msg
             |  offending line: $lineToPrint""".stripMargin)
    }
  }
}
