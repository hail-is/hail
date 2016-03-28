package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamples extends Command {

  def newOptions: Options = throw new UnsupportedOperationException

  def name = "annotatesamples"

  def description = "Annotate samples in current dataset"

  override def runCommand(state: State, options: Options): State = throw new UnsupportedOperationException

  override def lookup(args: Array[String]): (Command, Array[String]) = {
    val errorString =
      """parse error: expect one of the following:
        |  annotatesamples expr <args>
        |  annotatesamples tsv <args>
      """.stripMargin
    fatalIf(args.isEmpty, errorString)
    args.head match {
      case "expr" => (AnnotateSamplesExpr, args.tail)
      case "tsv" => (AnnotateSamplesTSV, args.tail)
      case _ => fatal(errorString)
    }
  }

  override def run(state: State, args: Array[String]): State = {
    val (c, newArgs) = lookup(args)
    c.run(state, newArgs)
  }

  override def run(state: State, options: Options): State = throw new UnsupportedOperationException
}
