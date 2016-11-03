package org.broadinstitute.hail.driver

import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps

object TDTCommand extends Command {

  def name = "tdt"

  def description = "Find transmitted and untransmitted variants; count per variant, nuclear family"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"), usage = "Annotation root, starting in `va'")
    var root: String = _
  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val ped = Pedigree.read(options.famFilename, state.hadoopConf, state.vds.sampleIds)
    state.copy(vds = TDT(state.vds, ped.completeTrios,
      Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)))
  }
}
