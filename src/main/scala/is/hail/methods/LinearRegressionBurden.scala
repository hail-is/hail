package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.keytable.KeyTable
import is.hail.stats._
import is.hail.utils._
import is.hail.expr._
import is.hail.variant._
import net.sourceforge.jdistlib.T
import org.apache.spark.sql.Row

object LinearRegressionBurden {

  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    aggExpr: String,
    yExpr: String,
    covExpr: Array[String]): (KeyTable, KeyTable) = {

    val (y, cov, completeSamplesIndex) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val completeSamples = completeSamplesIndex.map(vds.sampleIds)

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Aggregating variants by '$keyName' for $n samples...")

    val completeSamplesSet = completeSamples.toSet
    if (completeSamplesSet(keyName))
      fatal(s"Key name '$keyName' clashes with a sample name")

    val linregFields = LinearRegression.schema.fields.map(_.name).toSet
    if (linregFields(keyName))
      fatal(s"Key name '$keyName' clashes with reserved linreg columns $linregFields")

    def sampleKT = vds.filterSamples((s, sa) => completeSamplesSet(s))
      .aggregateBySamplePerVariantKey(keyName, variantKeys, aggExpr, singleKey)
      .cache()

    val keyType = sampleKT.fields(0).typ

    // d > 0 implies at least 1 sample
    val numericType = sampleKT.fields(1).typ

    if (!numericType.isInstanceOf[TNumeric])
      fatal(s"aggregate_expr type must be numeric, found $numericType")

    info(s"Running linear regression burden test for ${sampleKT.count} keys on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = sampleKT.hc.sc
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))

    val (linregSignature, merger) = TStruct(keyName -> keyType).merge(LinearRegressionModel.schema)
    
    val linregRDD = sampleKT.mapAnnotations { keyedRow =>
      val x = RegressionUtils.keyedRowToVectorDouble(keyedRow)
      merger(
        Row(keyedRow.get(0)),
        LinearRegressionModel.fit(x, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d)).asInstanceOf[Row]
    }
    val linregKT = new KeyTable(sampleKT.hc, linregRDD, signature = linregSignature, key = Array(keyName))

    (linregKT, sampleKT)
  }
}
