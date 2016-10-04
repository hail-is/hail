package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.VariantSampleMatrix
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object Read extends Command {
  def name = "read"

  def description = "Load .vds file(s) as the current dataset. If loading multiple files, they must have similar metadata."

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
    val wasSplit = vdses.head.wasSplit
    val reference = inputs(0)

    vdses.indices.tail.foreach { i =>
      val vds = vdses(i)
      val ids = vds.sampleIds
      val vas = vds.vaSignature
      val path = inputs(i)
      if (ids != sampleIds) {
        fatal(
          s"""cannot read datasets with different sample IDs or sample ordering
              |  IDs in reference file $reference: @1
              |  IDs in file $path: @2""".stripMargin, sampleIds, ids)
      } else if (wasSplit != vds.wasSplit) {
        fatal(
          s"""cannot combine split and unsplit datasets
              |  Reference file $reference split status: $wasSplit
              |  File $path split status: ${ vds.wasSplit }""".stripMargin)
      } else if (vas != vaSchema) {
        fatal(
          s"""cannot read datasets with different variant annotation schemata
              |  Schema in reference file $reference: @1
              |  Schema in file $path: @2""".stripMargin,
          vaSchema.toPrettyString(compact = true, printAttrs = true),
          vas.toPrettyString(compact = true, printAttrs = true)
        )
      }
    }

    if (vdses.length > 1)
      info(s"Using sample and global annotations from ${ inputs(0) }")

    state.copy(vds = vdses(0).copy(rdd = state.sc.union(vdses.map(_.rdd)).toOrderedRDD))
  }
}
