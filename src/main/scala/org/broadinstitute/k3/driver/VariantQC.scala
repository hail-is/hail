package org.broadinstitute.k3.driver

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object VariantQC extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }
  def newOptions = new Options

  def name = "variantqc"
  def description = "Compute per-variant QC metrics"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val methods: Array[Method] = Array(
      nCalledPer, nNotCalledPer,
      nHomRefPer, nHetPer, nHomVarPer
    )

    val derivedMethods: Array[DerivedMethod] = Array(
      nNonRefPer, rHetrozygosityPer, rHetHomPer, pHwePerVariant
    )

    val allMethods = methods ++ derivedMethods

    val methodIndex = methods.zipWithIndex.toMap

    val results: RDD[(Variant, Array[Any])] = vds.mapValuesWithKeys { (v, s, g) =>
      methods.map(_.mapWithKeys(v, s, g): Any)
    }.foldByVariant(methods.map(_.foldZeroValue)) { (x, y) =>
      methods.zipWith[Any, Any, Any](x, y, (m, xi, yi) =>
        m.fold(xi.asInstanceOf[m.T], yi.asInstanceOf[m.T]))
    }.mapValues { case values =>
      values ++ derivedMethods.map(_.map(MethodValues(methodIndex, values)))
    }

    writeTextFile(options.output + ".header", state.hadoopConf) { s =>
      val header = "Chrom" + "\t" + "Pos" + "\t" + "Ref" + "\t" + "Alt" + "\t" +
        allMethods.map(_.name).mkString("\t") + "\n"
      s.write(header)
    }

    results.map { case (v, a) =>
      (Array[Any](v.contig, v.start, v.ref, v.alt) ++ a).mkString("\t")
    }.saveAsTextFile(options.output)

    state
  }
}
