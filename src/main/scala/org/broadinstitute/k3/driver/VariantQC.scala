package org.broadinstitute.k3.driver

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object VariantQC extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "variantqc"

  def description = "Compute per-variant QC metrics"

  def results(vds: VariantDataset,
    methods: Array[AggregateMethod],
    derivedMethods: Array[DerivedMethod] = Array()): RDD[(Variant, Array[Any])] = {

    val methodIndex = methods.zipWithIndex.toMap

    val methodsBc = vds.sparkContext.broadcast(methods)

    vds
      .aggregateByVariantWithKeys(methods.map(_.aggZeroValue: Any))(
        (acc, v, s, g) => methodsBc.value.zipWith[Any, Any](acc, (m, acci) =>
            m.seqOpWithKeys(v, s, g, acci.asInstanceOf[m.T])),
        (x, y) => methodsBc.value.zipWith[Any, Any, Any](x, y, (m, xi, yi) =>
            m.combOp(xi.asInstanceOf[m.T], yi.asInstanceOf[m.T])))
      .mapValues(values => {
        val b = mutable.ArrayBuilder.make[Any]()
        values.foreach2[AggregateMethod](methodsBc.value, (v, m) => m.emit(v.asInstanceOf[m.T], b))
        val methodValues = MethodValues(methodIndex, values)
        derivedMethods.foreach(_.emit(methodValues, b))
        b.result()
      })
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val methods: Array[AggregateMethod] = Array(
      nCalledPer, nNotCalledPer,
      nHomRefPer, nHetPer, nHomVarPer,
      AlleleBalancePer, dpStatCounterPer, dpStatCounterPerGenotype, gqStatCounterPer, gqStatCounterPerGenotype
    )

    val derivedMethods: Array[DerivedMethod] = Array(
      minorAlleleFrequencyPer, nNonRefPer, rHeterozygosityPer, rHetHomVarPer, pHwePerVariant
    )

    val r = results(vds, methods, derivedMethods)

    val allMethods = methods ++ derivedMethods

    writeTextFile(options.output + ".header", state.hadoopConf) { s =>
      val header = "Chrom" + "\t" + "Pos" + "\t" + "Ref" + "\t" + "Alt" + "\t" +
        allMethods.map(_.name).filter(_ != null).mkString("\t") + "\n"
      s.write(header)
    }

    val output = options.output

    hadoopDelete(output, vds.sparkContext.hadoopConfiguration, true)
    r.map { case (v, a) =>
      (Array[Any](v.contig, v.start, v.ref, v.alt) ++ a).map(toTSVString).mkString("\t")
    }.saveAsTextFile(output)

    state
  }
}
