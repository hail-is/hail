package org.broadinstitute.k3.driver

import org.apache.spark.broadcast.Broadcast
import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object SampleQC extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "sampleqc"

  def description = "Compute per-sample QC metrics"

  def results(vds: VariantDataset,
    methods: Array[AggregateMethod],
    derivedMethods: Array[DerivedMethod] = Array()): Map[Int, Array[Any]] = {
    val methodIndex = methods.zipWithIndex.toMap

    val methodsBc = vds.sparkContext.broadcast(methods)

    vds
      .aggregateBySampleWithKeys(methods.map(_.aggZeroValue: Any))(
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
    val sc = state.sc
    val vds = state.vds

    val singletons: Broadcast[Set[Variant]] = sc.broadcast(sSingletonVariants(vds))

    val methods: Array[AggregateMethod] = Array(
      nCalledPer, nNotCalledPer,
      nHomRefPer, nHetPer, nHomVarPer, AlleleBalancePer,
      nSNPPerSample, nInsertionPerSample, nDeletionPerSample,
      new nSingletonPerSample(singletons), nTransitionPerSample, nTransversionPerSample,
      dpStatCounterPer, dpStatCounterPerGenotype, gqStatCounterPer, gqStatCounterPerGenotype
    )

    val derivedMethods: Array[DerivedMethod] = Array(
      nNonRefPer, rTiTvPerSample, rHetHomVarPer, rDeletionInsertionPerSample
    )

    val r = results(vds, methods, derivedMethods)
    writeTextFile(options.output, state.hadoopConf) { s =>
      val allMethods = methods ++ derivedMethods
      val header = "sampleID" + "\t" + allMethods.map(_.name).filter(_ != null).mkString("\t") + "\n"
      s.write(header)

      for (i <- r.keys)
        s.write(vds.sampleIds(i) + "\t" + r(i).map(toTSVString).mkString("\t") + "\n")
    }

    state
  }
}
