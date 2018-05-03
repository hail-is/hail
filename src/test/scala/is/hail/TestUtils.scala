package is.hail

import java.net.URI
import java.nio.file.{Files, Paths}

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import is.hail.annotations.Annotation
import is.hail.expr.types.{TFloat64, TString}
import is.hail.linalg.BlockMatrix
import is.hail.methods.{KinshipMatrix, SplitMulti}
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.SparkException

object TestUtils {

  import org.scalatest.Assertions._

  def interceptFatal(regex: String)(f: => Any) {
    val thrown = intercept[HailException](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    if (!p)
      println(
        s"""expected fatal exception with pattern `$regex'
           |  Found: ${ thrown.getMessage } """.stripMargin)
    assert(p)
  }

  def interceptSpark(regex: String)(f: => Any) {
    val thrown = intercept[SparkException](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    if (!p)
      println(
        s"""expected fatal exception with pattern `$regex'
           |  Found: ${ thrown.getMessage } """.stripMargin)
    assert(p)
  }

  def interceptAssertion(regex: String)(f: => Any) {
    val thrown = intercept[AssertionError](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    if (!p)
      println(
        s"""expected assertion error with pattern `$regex'
           |  Found: ${ thrown.getMessage } """.stripMargin)
    assert(p)
  }

  def assertVectorEqualityDouble(A: Vector[Double], B: Vector[Double], tolerance: Double = utils.defaultTolerance) {
    assert(A.size == B.size)
    assert((0 until A.size).forall(i => D_==(A(i), B(i), tolerance)))
  }

  def assertMatrixEqualityDouble(A: Matrix[Double], B: Matrix[Double], tolerance: Double = utils.defaultTolerance) {
    assert(A.rows == B.rows)
    assert(A.cols == B.cols)
    assert((0 until A.rows).forall(i => (0 until A.cols).forall(j => D_==(A(i, j), B(i, j), tolerance))))
  }

  def isConstant(A: Vector[Int]): Boolean = {
    (0 until A.length - 1).foreach(i => if (A(i) != A(i + 1)) return false)
    true
  }

  def removeConstantCols(A: DenseMatrix[Int]): DenseMatrix[Int] = {
    val data = (0 until A.cols).flatMap { j =>
      val col = A(::, j)
      if (TestUtils.isConstant(col))
        Array[Int]()
      else
        col.toArray
    }.toArray

    val newCols = data.length / A.rows
    new DenseMatrix(A.rows, newCols, data)
  }

  // missing is -1
  def vdsToMatrixInt(vds: MatrixTable): DenseMatrix[Int] =
    new DenseMatrix[Int](
      vds.numCols,
      vds.countRows().toInt,
      vds.typedRDD[Locus].map(_._2._2.map { g =>
        Genotype.call(g)
          .map(Call.nNonRefAlleles)
          .getOrElse(-1)
      }).collect().flatten)

  // missing is Double.NaN
  def vdsToMatrixDouble(vds: MatrixTable): DenseMatrix[Double] =
    new DenseMatrix[Double](
      vds.numCols,
      vds.countRows().toInt,
      vds.rdd.map(_._2._2.map { g =>
        Genotype.call(g)
          .map(Call.nNonRefAlleles)
          .map(_.toDouble)
          .getOrElse(Double.NaN)
      }).collect().flatten)

  def unphasedDiploidGtIndicesToBoxedCall(m: DenseMatrix[Int]): DenseMatrix[BoxedCall] = {
    m.map(g => if (g == -1) null: BoxedCall else Call2.fromUnphasedDiploidGtIndex(g): BoxedCall)
  }

  def indexedSeqBoxedDoubleEquals(tol: Double)
    (xs: IndexedSeq[java.lang.Double], ys: IndexedSeq[java.lang.Double]): Boolean =
    (xs, ys).zipped.forall { case (x, y) =>
      if (x == null || y == null)
        x == null && y == null
      else
        D_==(x.doubleValue(), y.doubleValue(), tolerance = tol)
    }

  def keyTableBoxedDoubleToMap[T](kt: Table): Map[T, IndexedSeq[java.lang.Double]] =
    kt.collect().map { r =>
      val s = r.toSeq
      s.head.asInstanceOf[T] -> s.tail.map(_.asInstanceOf[java.lang.Double]).toIndexedSeq
    }.toMap

  def matrixToString(A: DenseMatrix[Double], separator: String): String = {
    val sb = new StringBuilder
    for (i <- 0 until A.rows) {
      for (j <- 0 until A.cols) {
        if (j == (A.cols - 1))
          sb.append(A(i, j))
        else {
          sb.append(A(i, j))
          sb.append(separator)
        }
      }
      sb += '\n'
    }
    sb.result()
  }

  def fileHaveSameBytes(file1: String, file2: String): Boolean =
    Files.readAllBytes(Paths.get(URI.create(file1))) sameElements Files.readAllBytes(Paths.get(URI.create(file2)))

  def splitMultiHTS(mt: MatrixTable): MatrixTable = {
    if (!mt.entryType.isOfType(Genotype.htsGenotypeType))
      fatal(s"split_multi: genotype_schema must be the HTS genotype schema, found: ${ mt.entryType }")
    val pl = """if (isDefined(g.PL))
    range(3).map(i => range(g.PL.size()).filter(j => downcode(UnphasedDiploidGtIndexCall(j), aIndex) == UnphasedDiploidGtIndexCall(i)).map(j => g.PL[j]).min())
    else
    NA: Array[Int]"""
    SplitMulti(mt, "va.aIndex = aIndex, va.wasSplit = wasSplit",
      s"""g.GT = downcode(g.GT, aIndex),
      g.AD = if (isDefined(g.AD))
          let sum = g.AD.sum() and adi = g.AD[aIndex] in [sum - adi, adi]
        else
          NA: Array[Int],
          g.DP = g.DP,
      g.PL = $pl,
      g.GQ = gqFromPL($pl)""")
  }
  
  // !useHWE: mean 0, norm exactly sqrt(n), variance 1
  // useHWE: mean 0, norm approximately sqrt(m), variance approx. m / n
  // missing gt are mean imputed, constant variants return None, only HWE uses nVariants
  def normalizedHardCalls(view: HardCallView, nSamples: Int, useHWE: Boolean = false, nVariants: Int = -1): Option[Array[Double]] = {
    require(!(useHWE && nVariants == -1))
    val vals = Array.ofDim[Double](nSamples)
    var nMissing = 0
    var sum = 0
    var sumSq = 0

    var row = 0
    while (row < nSamples) {
      view.setGenotype(row)
      if (view.hasGT) {
        val gt = Call.unphasedDiploidGtIndex(view.getGT)
        vals(row) = gt
        (gt: @unchecked) match {
          case 0 =>
          case 1 =>
            sum += 1
            sumSq += 1
          case 2 =>
            sum += 2
            sumSq += 4
        }
      } else {
        vals(row) = -1
        nMissing += 1
      }
      row += 1
    }

    val nPresent = nSamples - nMissing
    val nonConstant = !(sum == 0 || sum == 2 * nPresent || sum == nPresent && sumSq == nPresent)

    if (nonConstant) {
      val mean = sum.toDouble / nPresent
      val stdDev = math.sqrt(
        if (useHWE)
          mean * (2 - mean) * nVariants / 2
        else {
          val meanSq = (sumSq + nMissing * mean * mean) / nSamples
          meanSq - mean * mean
        })

      val gtDict = Array(0, -mean / stdDev, (1 - mean) / stdDev, (2 - mean) / stdDev)
      var i = 0
      while (i < nSamples) {
        vals(i) = gtDict(vals(i).toInt + 1)
        i += 1
      }

      Some(vals)
    } else
      None
  }
  
  def computeRRM(hc: HailContext, vds: MatrixTable): KinshipMatrix = {
    var mt = vds
    mt = mt.selectEntries("{gt: g.GT.nNonRefAlleles()}")
      .annotateRowsExpr("AC" -> "AGG.map(g => g.gt).sum()",
                        "ACsq" -> "AGG.map(g => g.gt * g.gt).sum()",
                        "nCalled" -> "AGG.filter(g => isDefined(g.gt)).count().toInt32")
      .filterRowsExpr("(va.AC > 0) && (va.AC < 2 * va.nCalled) && ((va.AC != va.nCalled) || (va.ACsq != va.nCalled))")
    
    val (nVariants, nSamples) = mt.count()
    require(nVariants > 0, "Cannot run RRM: found 0 variants after filtering out monomorphic sites.")

    mt = mt.annotateGlobal(nSamples.toDouble, TFloat64(), "nSamples")
      .annotateRowsExpr("meanGT" -> "va.AC.toFloat64 / va.nCalled.toFloat64")
    mt = mt.annotateRowsExpr("stdDev" ->
      "((va.ACsq.toFloat64 + (global.nSamples - va.nCalled.toFloat64).toFloat64 * va.meanGT * va.meanGT) / global.nSamples - va.meanGT * va.meanGT).sqrt()")

    val normalizedGT = "let norm_gt = (g.gt.toFloat64 - va.meanGT) / va.stdDev in if (isDefined(norm_gt)) norm_gt else 0.0"
    val path = TempDir(hc.hadoopConf).createTempFile()
    mt.selectEntries(s"{x: $normalizedGT}").writeBlockMatrix(path, "x")
    val X = BlockMatrix.read(hc, path)
    val rrm = X.transpose().dot(X).scalarDiv(nVariants).toIndexedRowMatrix()

    KinshipMatrix(vds.hc, TString(), rrm, vds.stringSampleIds.map(s => s: Annotation).toArray, nVariants)
  }

  def exportPlink(mt: MatrixTable, path: String): Unit = {
    mt.selectCols("""{fam_id: "0", id: sa.s, mat_id: "0", pat_id: "0", is_female: "0", pheno: "NA"}""")
      .annotateRowsExpr(
        "varid" -> """let l = va.locus and a = va.alleles in [l.contig, str(l.position), a[0], a[1]].mkString(":")""",
        "pos_morgan" -> "0")
      .exportPlink(path)
  }
}
