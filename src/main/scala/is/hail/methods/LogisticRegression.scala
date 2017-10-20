package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object LogisticRegression {

  def apply[RPK, RK, T >: Null](vsm: VariantSampleMatrix[RPK, RK, T],
    test: String,
    yExpr: String,
    xExpr: String,
    covExpr: Array[String],
    root: String
  )(implicit tct: ClassTag[T]): VariantSampleMatrix[RPK, RK, T] = {
    val ec = vsm.matrixType.genotypeEC
    val xf = RegressionUtils.parseXExpr(xExpr, ec)

    val logRegTest = LogisticRegressionTest.tests.getOrElse(test,
      fatal(s"Supported tests are ${ LogisticRegressionTest.tests.keys.mkString(", ") }, got: $test"))

    val (y, cov, completeSampleIndex) = RegressionUtils.getPhenoCovCompleteSamples(vsm, yExpr, covExpr)
    val completeSamples = completeSampleIndex.map(vsm.sampleIds)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vsm.sampleIds.map(completeSamplesSet).toArray

    if (!y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all present values equal to 0 or 1")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running $test logistic regression on $n samples with $k ${ plural(k, "covariate") } including intercept...")

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

    val sampleMaskBc = sc.broadcast(sampleMask)
    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(y)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest)

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vsm.insertVA(logRegTest.schema, pathVA)

    val newRDD = vsm.rdd.mapPartitionsPreservingPartitioning { it =>
      val missingSamples = new ArrayBuilder[Int]()

      val X = XBc.value.copy
      it.map { case row@(v, (va, gs)) =>
        RegressionUtils.exprDosages(X(::, -1),
          localGlobalAnnotationBc.value, sampleIdsBc.value, sampleAnnotationsBc.value, row,
          ec, xf,
          completeSampleIndexBc.value, missingSamples)

        val logregAnnot = logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation
        val newAnnotation = inserter(va, logregAnnot)
        assert(newVAS.typeCheck(newAnnotation))
        (v, (newAnnotation, gs))
      }
    }

    vsm.copy(vaSignature = newVAS,
      rdd = newRDD)
  }
}

case class LogisticRegression(rdd: RDD[(Variant, Annotation)], schema: Type)