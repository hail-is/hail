package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T
import org.apache.spark.sql.Row

object LinearRegressionBurden {

  def apply(vds: VariantDataset, keyName: String, variantKeySetVA: String, aggregateWith: String, genotypeExpr: String, ySA: String, covSA: Array[String], dropSamples: Boolean): KeyTable = {

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, ySA, covSA)

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linreg_burden, aggregated by key $keyName using $aggregateWith, on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val completeSamplesSet = completeSamples.toSet

    if (completeSamplesSet(keyName))
      fatal(s"Sample name conflicts with the key name $keyName")

    if (!dropSamples) {
      val conflicts = completeSamplesSet.intersect(LinearRegression.`type`.fields.map(x => x.name).toSet)
      if (conflicts.nonEmpty)
        fatal(s"Sample names conflict with these reserved statistical names: ${conflicts.mkString(", ")}")
    }

    val (variantKeysType, variantKeysQuerier) = vds.queryVA(variantKeySetVA)

    if (variantKeysType != TSet(TString))
      fatal(s"Variant keys must be of type Set(String), got $variantKeysType")

    val aggExpr = completeSamples.map(s => s"`$s` = `$s`.$aggregateWith").mkString(", ")

    val kt: KeyTable = vds.filterSamples { case (s, sa) => completeSamplesSet(s) }
      .filterVariants { case (v, va, gs) => variantKeysQuerier(va).asInstanceOf[Set[String]].nonEmpty }
      .makeKT(s"$keyName = $variantKeySetVA", s"`` = $genotypeExpr", Array[String]())
      .explode(keyName)
      .aggregate(s"$keyName = $keyName", aggExpr)

    val emptyStats = Annotation.emptyIndexedSeq(4)

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = kt.hc.sc
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))

    val newRDD = kt.mapAnnotations { a =>
      val (keySeq, dataX) = a.asInstanceOf[Row].toSeq.splitAt(1)
      val key = keySeq.head.asInstanceOf[String]

      RegressionUtils.denseStats(dataX.asInstanceOf[Seq[java.lang.Number]], y) match {
        case Some((x, xx, xy)) =>
          val qtx = QtBc.value * x
          val qty = QtyBc.value
          val xxp: Double = xx - (qtx dot qtx)
          val xyp: Double = xy - (qtx dot qty)
          val yyp: Double = yypBc.value

          val b = xyp / xxp
          val se = math.sqrt((yyp / xxp - b * b) / d)
          val t = b / se
          val p = 2 * T.cumulative(-math.abs(t), d, true, false)

          if (dropSamples)
            Annotation(key, b, se, t, p)
          else
            Row.fromSeq(Seq(key, b, se, t, p) ++ dataX): Annotation
        case None =>
          if (dropSamples)
            Annotation(key +: emptyStats)
          else
            Row.fromSeq((key +: emptyStats) ++ dataX): Annotation
      }
    }

    def newSignature =
      if (dropSamples)
        TStruct((keyName -> TString) +: LinearRegression.`type`.fields.map(x => x.name -> x.typ): _*)
      else
        TStruct(((keyName -> TString) +: LinearRegression.`type`.fields.map(x => x.name -> x.typ))
          ++ kt.fields.tail.map(x => x.name -> x.typ): _*)

    new KeyTable(kt.hc, newRDD, signature = newSignature, keyNames = Array(keyName))
  }
}