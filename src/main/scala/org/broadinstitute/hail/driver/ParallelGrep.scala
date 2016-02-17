package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}


object ParallelGrep extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "grep this file")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--regex"), usage = "apply this regex")
    var regex: String = _

    @Args4jOption(required = false, name = "-s", aliases = Array("--stop"), usage = "stop after one match")
    var stop: Boolean = false

    @Args4jOption(required = false, name = "-o", aliases = Array("--output"), usage = "write matches to file")
    var out: String = _
  }

  def newOptions = new Options

  def name = "grep"

  def description = "Grep a big file, like, really fast"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {

    val regex = options.regex.r
    val stop = options.stop

    val rdd = state.sc.textFile(options.input)
      .filter(line => regex.findFirstIn(line).isDefined)


    if (options.stop) {
      val toPrint = rdd.take(1).head
      options.out match {
        case null => println(rdd.take(1).head)
        case output => writeTextFile(output, state.sc.hadoopConfiguration){dos => dos.write(toPrint)}
      }
    }
    else {
      options.out match {
        case null => rdd.foreach(println)
        case output => rdd.writeTable(output)
      }
    }

    state
  }

}
