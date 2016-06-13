package org.broadinstitute.hail.driver

import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps

object TDTCommand extends Command {

  def name = "tdt"

  def description = "Find transmitted and untransmitted variants; count per variant, nuclear family"

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-o", aliases = Array("--output"), usage = "Output root filename")
    var output: String = _

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
    val resultsRDD = TDT(state.vds, ped.completeTrios).persist(StorageLevel.MEMORY_AND_DISK)

    val (finalType, inserter) = state.vds.insertVA(TDT.schema, Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))

    val newRDD = state.vds.rdd.zipPartitions(resultsRDD, preservesPartitioning = true) { case (it1, it2) =>
      it1.sortedLeftJoinDistinct(it2).map { case (v, ((va, gs), tdtResult)) => (v, (inserter(va, tdtResult.map(_.toAnnotation)), gs)) }
    }.toOrderedRDD(_.locus)

    Option(options.output).foreach { filename =>
      resultsRDD.map { case (v, tdtResult) =>
        v.contig + "\t" + v.start + "\t" + v.ref + "\t" + v.alt + "\t" + tdtResult.nTransmitted + "\t" + tdtResult.nUntransmitted + "\t" + tdtResult.chiSquare
      }
        .writeTable(filename + "VariantRes.tdt.txt",
          Some("CHROM\tPOSITION\tREF\tALT\tTransmitted\tUntransmitted\tChi-Square"))
    }

    state.copy(vds = state.vds.copy(rdd = newRDD, vaSignature = finalType))
  }
}
