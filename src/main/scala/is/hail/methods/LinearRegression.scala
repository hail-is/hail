package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._

object LinearRegression {
  def apply(vds: VariantDataset, yExpr: String, covExpr: Array[String], root: String, useDosages: Boolean, minAC: Int, minAF: Double, fields: Array[String]): VariantDataset = {
    require(vds.wasSplit)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray
    val completeSampleIndex = (0 until vds.nSamples)
      .filter(i => completeSamplesSet(vds.sampleIds(i)))
      .toArray

    val n = y.size
    val nFields = fields.length
    
    val k = cov.cols + (if (nFields == 0) 1 else nFields)
    val d = n - k

    if (minAC < 1)
      fatal(s"Minumum alternate allele count must be a positive integer, got $minAC")
    if (minAF < 0 || minAF > 1)
      fatal(s"Minumum alternate allele frequency must lie in [0.0, 1.0], got $minAF")
    val combinedMinAC = math.max(minAC, (math.ceil(2 * n * minAF) + 0.5).toInt)

    if (d < 1)
      fatal(s"$n samples with ${if (nFields == 0) 1 else nFields} ${ plural(nFields, "field") } and ${cov.cols} ${ plural(cov.cols, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linear regression on $n samples with ${if (nFields == 0) 1 else nFields} ${ plural(nFields, "field") } and ${cov.cols} ${ plural(cov.cols, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y
    
    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))
    
    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val schema = if (fields.isEmpty) LinearRegressionModel.schema else LinearRegressionModel.schemaMultiField
    val (newVAS, inserter) = vds.insertVA(schema, pathVA)

    if (fields.isEmpty) {
      vds.mapAnnotations { case (v, va, gs) =>
        val linregAnnot = {
          val (x: Vector[Double], ac) =
            if (!useDosages) // replace by hardCalls in 0.2, with ac post-imputation
              RegressionUtils.hardCallsWithAC(gs, n, sampleMaskBc.value)
            else {
              val x = RegressionUtils.dosages(gs, completeSampleIndexBc.value)
              (x, sum(x))
            }
    
          // constant checking to be removed in 0.2
          val nonConstant = useDosages || !RegressionUtils.constantVector(x)
    
          if (ac >= combinedMinAC && nonConstant)
            LinearRegressionModel.fit(x, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d).toAnnotation
          else
            null
        }
  
        val newAnnotation = inserter(va, linregAnnot)
        assert(newVAS.typeCheck(newAnnotation))
        newAnnotation
      }.copy(vaSignature = newVAS)
    } else {
      val vas = vds.vaSignature
      val sas = vds.saSignature

      val symTab = Map(
        "v" -> (0, TVariant),
        "va" -> (1, vas),
        "s" -> (2, TString),
        "sa" -> (3, sas),
        "g" -> (4, TGenotype),
        "global" -> (5, vds.globalSignature))

      val ec = EvalContext(symTab)
      ec.set(5, vds.globalAnnotation) // is this shared correctly by workers?

      val samplesIds = vds.sampleIds // filter before broadcast?
      val sampleAnnotations = vds.sampleAnnotations

      val sampleIdsBc = sc.broadcast(samplesIds)
      val sampleAnnotationsBc = sc.broadcast(sampleAnnotations)

      val (types, fs) = Parser.parseExprs(fields.mkString(","), ec)
     
      val aToDouble = (types, fields).zipped.map(RegressionUtils.toDouble)
      
      vds.mapAnnotations { case (v, va, gs) =>
        val linregAnnot = {
          ec.set(0, v)
          ec.set(1, va)

          val data = Array.ofDim[Double](n * nFields)
          val sums = Array.ofDim[Double](nFields)
          val nMissings = Array.ofDim[Int](nFields)
          val gsIter = gs.iterator

          val missingRows = new ArrayBuilder[Int]()
          val missingCols = new ArrayBuilder[Int]()

          var r = 0
          var c = 0
          var i = 0
          while (i < sampleMask.length) {
            val g = gsIter.next()
            if (sampleMask(i)) {
              ec.set(2, sampleIdsBc.value(i))
              ec.set(3, sampleAnnotationsBc.value(i))
              ec.set(4, g)

              val row = (fs(), aToDouble).zipped.map { (e, td) => if (e == null) Double.NaN else td(e) }

              c = 0
              while (c < nFields) {
                val e = row(c)
                if (!e.isNaN)
                  sums(c) += e
                else {
                  missingRows += r
                  missingCols += c
                  nMissings(c) += 1
                }
                data(c * n + r) = e
                c += 1
              }
              r += 1
            }
            i += 1
          }

          if (!nMissings.contains(n)) {
            val means = (sums, nMissings).zipped.map { case (sum, nMissing) => sum / (n - nMissing) }
            
            i = 0
            while (i < missingRows.length) {
              val c = missingCols(i)
              data(c * nFields + missingRows(i)) = means(c)
              i += 1
            }

            val X = new DenseMatrix[Double](n, nFields, data)
            LinearRegressionModel.fit(X, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d).toAnnotation
          } else
            null
        }

        val newAnnotation = inserter(va, linregAnnot)
        assert(newVAS.typeCheck(newAnnotation))
        newAnnotation
      }.copy(vaSignature = newVAS)
    }
  }
}