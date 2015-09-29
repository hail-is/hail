package org.broadinstitute.k3.driver

import org.apache.spark.broadcast.Broadcast
import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object SampleQC extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }
  def newOptions = new Options

  def name = "sampleqc"
  def description = "Compute per-sample QC metrics"

  def run(state: State, options: Options): State = {
    val sc = state.sc
    val vds = state.vds

    val singletons: Broadcast[Set[Variant]] = sc.broadcast(sSingletonVariants(vds))

    val methods: Array[Method] = Array(
      nCalledPer, nNotCalledPer,
      nHomRefPer, nHetPer, nHomVarPer,
      nSNPPerSample, nInsertionPerSample, nDeletionPerSample,
      new nSingletonPerSample(singletons), nTransitionPerSample, nTransversionPerSample
    )

    val derivedMethods: Array[DerivedMethod] = Array(
      nNonRefPer, rTiTvPerSample, rHetHomPer, rDeletionInsertionPerSample
    )

    val methodIndex = methods.zipWithIndex.toMap

    val methodsBc = vds.sparkContext.broadcast(methods)

    val mapStep = vds.mapValuesWithKeys { (v, s, g) =>
      methodsBc.value.map(_.mapWithKeys(v, s, g): Any)
    }

    mapStep.expand().collect().foreach(t => println("" + t._1 + " " + t._2 + " " + t._3.mkString(",")))

    val foldStep = mapStep.foldBySample(methods.map(_.foldZeroValue)) { (acc, x) =>
      methodsBc.value.zipWith[Any, Any, Any](acc, x, (m, xi, yi) =>
        m.fold(xi.asInstanceOf[m.T], yi.asInstanceOf[m.T]))

      /*
      for (i <- methodsBc.value.indices) {
        val m = methodsBc.value(i)
        acc(i) = m.fold(acc(i).asInstanceOf[m.T], x(i).asInstanceOf[m.T])
      }
      acc
      */
    }

    foldStep.foreach(t => println("" + t._1 + t._2.mkString(",")))

    val sampleResults: Map[Int, Array[Any]] = vds.mapValuesWithKeys { (v, s, g) =>
      methodsBc.value.map(_.mapWithKeys(v, s, g): Any)
    }.foldBySample(methods.map(_.foldZeroValue)) { (acc, x) =>
      methodsBc.value.zipWith[Any, Any, Any](acc, x, (m, acci, xi) =>
        m.fold(acci.asInstanceOf[m.T], xi.asInstanceOf[m.T]))
      /*
      for (i <- methodsBc.value.indices) {
        val m = methodsBc.value(i)
        acc(i) = m.fold(acc(i).asInstanceOf[m.T], x(i).asInstanceOf[m.T])
      }
      acc
      */
    }.mapValues { case values =>
      values ++ derivedMethods.map(_.map(MethodValues(methodIndex, values)))
    }

    writeTextFile(options.output, state.hadoopConf) { s =>
      val allMethods = methods ++ derivedMethods
      val header = "sampleID" + "\t" + allMethods.map(_.name).mkString("\t") + "\n"
      s.write(header)

      for ((id, i) <- vds.sampleIds.zipWithIndex) {
        s.write(id + "\t" + sampleResults(i).mkString("\t") + "\n")
      }
    }

    state
  }
}
