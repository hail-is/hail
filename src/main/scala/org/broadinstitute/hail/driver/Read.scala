package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.VariantSampleMatrix
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object Read extends Command {
  def name = "read"

  def description = "Load file .vds as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "-i", aliases = Array("--input"), usage = "Input file (deprecated)")
    var input: Boolean = false

    @Args4jOption(name = "--skip-genotypes", usage = "Don't load genotypes")
    var skipGenotypes: Boolean = false

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val input = options.input

    val inputs = state.hadoopConf.globAll(options.arguments.asScala.toIndexedSeq)
    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    val vdses = inputs.map(input => VariantSampleMatrix.read(state.sqlContext, input, options.skipGenotypes))

    val sampleIds = vdses.head.sampleIds
    val vaSchema = vdses.head.vaSignature
    val reference = inputs(0)

    vdses.indices.tail.foreach { i =>
      val ids = vdses(i).sampleIds
      val vas = vdses(i).vaSignature
      val path = inputs(i)
      if (ids != sampleIds) {
        log.error(s"IDs in reference file $reference: ${ sampleIds.mkString(", ") }")
        log.error(s"IDs in problem file $path: ${ ids.mkString(", ") }")
        fatal(
          s"""cannot read datasets with different sample IDs or sample ordering
              |  Expected IDs read from $reference
              |  Mismatch in file $path
              |  See log for full sample ID readout""".stripMargin)
      } else if (vas != vaSchema) {
        log.error(s"variant annotation schema in reference file $reference: ${ vaSchema.toPrettyString(compact = true, printAttrs = true) }")
        log.error(s"variant annotation schema in problem file $path: ${ vas.toPrettyString(compact = true, printAttrs = true) }")
        fatal(
          s"""cannot read datasets with different variant annotation schemata
              |  Expected schema read from $reference
              |  Mismatch in file $path
              |  See log for full schema readout""".stripMargin)
      }
    }

    if (vdses.length > 1)
      info(s"Using sample and global annotations from ${ inputs(0) }")

    state.copy(vds = vdses(0).copy(rdd = state.sc.union(vdses.map(_.rdd)).toOrderedRDD))
  }
}
