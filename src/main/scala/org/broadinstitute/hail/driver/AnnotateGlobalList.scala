package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable
import scala.io.Source

object AnnotateGlobalList extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Path to file")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `global'")
    var root: String = _

    @Args4jOption(name = "--as-set", usage = "Store the list as a set instead of an array (to do contains operations)")
    var asSet: Boolean = false
  }

  def newOptions = new Options

  def name = "annotateglobal list"

  def description = "Annotate global table from a text file"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val textList = readFile(options.input, state.hadoopConf) { in =>
      Source.fromInputStream(in)
        .getLines()
        .toArray
    }

    val (sig, toInsert) =
      if (options.asSet)
        (TSet(TString), textList.toSet)
      else
        (TArray(TString), textList: IndexedSeq[String])

    val path = Parser.parseAnnotationRoot(options.root, "global")

    val (newGlobalSig, inserter) = vds.insertGlobal(sig, path)

    state.copy(vds = vds.copy(
      globalAnnotation = inserter(vds.globalAnnotation, Some(toInsert)),
      globalSignature = newGlobalSig))
  }
}

