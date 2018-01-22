package is.hail.methods

import breeze.linalg._
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object LogisticRegression {

  def apply(vsm: MatrixTable,
    test: String,
    yExpr: String,
    xExpr: String,
    covExpr: Array[String],
    root: String): MatrixTable = {
    val logRegTest = LogisticRegressionTest.tests.getOrElse(test,
      fatal(s"Supported tests are ${ LogisticRegressionTest.tests.keys.mkString(", ") }, got: $test"))

    val ec = vsm.matrixType.genotypeEC
    val xf = RegressionUtils.parseExprAsDouble(xExpr, ec)

    val (y, cov, completeSampleIndex) = RegressionUtils.getPhenoCovCompleteSamples(vsm, yExpr, covExpr)

    if (!y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all present values equal to 0 or 1")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and ${ k + 1 } ${ plural(k, "covariate") } (including x and intercept) implies $d degrees of freedom.")

    info(s"logreg: running $test logistic regression on $n samples for response variable y,\n"
       + s"    with input variable x, intercept, and ${ k - 1 } additional ${ plural(k - 1, "covariate") }...")

    val nullModel = new LogisticRegressionModel(cov, y)
    val nullFit = nullModel.fit()

    if (!nullFit.converged)
      fatal("Failed to fit logistic regression null model (MLE with covariates only): " + (
        if (nullFit.exploded)
          s"exploded at Newton iteration ${ nullFit.nIter }"
        else
          "Newton iteration failed to converge"))

    val sc = vsm.sparkContext

    val localGlobalAnnotationBc = sc.broadcast(vsm.globalAnnotation)
    val sampleIdsBc = vsm.sampleIdsBc
    val sampleAnnotationsBc = vsm.sampleAnnotationsBc

    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(y)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest)

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    val (newRVDType, inserter) = vsm.rdd2.typ.insert(logRegTest.schema, "va" :: pathVA)
    val newVAType = newRVDType.rowType.fieldType(2).asInstanceOf[TStruct]

    val localRowType = vsm.rowType
    val newRVD = vsm.rdd2.mapPartitionsPreservesPartitioning(newRVDType) { it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      val missingSamples = new ArrayBuilder[Int]()

      val X = XBc.value.copy
      it.map { rv =>
        val ur = new UnsafeRow(localRowType, rv)
        val v = ur.get(1)
        val va = ur.get(2)
        val gs = ur.getAs[IndexedSeq[Annotation]](3)

        RegressionUtils.inputVector(X(::, -1),
          localGlobalAnnotationBc.value, sampleIdsBc.value, sampleAnnotationsBc.value, (v, (va, gs)),
          ec, xf,
          completeSampleIndexBc.value, missingSamples)

        val logregAnnot = logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation

        rvb.set(rv.region)
        rvb.start(newRVDType.rowType)
        inserter(rv.region, rv.offset, rvb, () =>
          rvb.addAnnotation(logRegTest.schema, logregAnnot))

        rv2.set(rv.region, rvb.end())
        rv2
      }
    }

    vsm.copy2(vaSignature = newVAType,
      rdd2 = newRVD)
  }
}

case class LogisticRegression(rdd: RDD[(Variant, Annotation)], schema: Type)
