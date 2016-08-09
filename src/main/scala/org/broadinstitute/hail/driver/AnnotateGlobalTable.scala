package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.{TextTableOptions, TextTableReader}
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateGlobalTable extends Command with JoinAnnotator {

  class Options extends BaseOptions with TextTableOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "Path to file")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `global'")
    var root: String = _
  }

  def newOptions = new Options

  def name = "annotateglobal table"

  def description = "Loads a text file by column as an `Array[Struct]` in global annotations."

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val path = Parser.parseAnnotationRoot(options.root, Annotation.GLOBAL_HEAD)

    val (struct, rdd) = TextTableReader.read(state.sc, Array(options.input), options.config)
    val arrayType = TArray(struct)

    val (finalType, inserter) = vds.insertGlobal(arrayType, Parser.parseAnnotationRoot(options.root, Annotation.GLOBAL_HEAD))

    val table = rdd
      .map(_.value)
      .collect(): IndexedSeq[Annotation]

    state.copy(vds = vds.copy(
      globalAnnotation = inserter(vds.globalAnnotation, Some(table)),
      globalSignature = finalType))
  }
}
