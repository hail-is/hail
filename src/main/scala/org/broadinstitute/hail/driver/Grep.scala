package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption, Argument}
import scala.collection.JavaConverters._

object Grep extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-m", aliases = Array("--max-count"), usage = "Stop after <num> matches")
    var max: Int = 10

    @Argument(required = true, usage = "<regex> <files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def name = "grep"

  def description = "Grep a big file, like, really fast"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val sc = state.sc
    val args = options.arguments.asScala

    if (args.length < 2)
      fatal("usage: <regex> <files...>")

    val regex = args.head.r
    val files = args.tail

    sc.union(files.map { f =>
      sc.textFile(f)
    }).filter(line => regex.findFirstIn(line).isDefined)
      .take(options.max)
      .foreach(println)

    state
  }

}
