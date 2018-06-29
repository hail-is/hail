package is.hail

import java.net.URI
import java.nio.file.{Files, Paths}

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.linalg.BlockMatrix
import is.hail.methods.{KinshipMatrix, SplitMulti}
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.SparkException
import org.apache.spark.sql.Row

object TestUtils {

  import org.scalatest.Assertions._

  def interceptException[E <: Throwable : Manifest](regex: String)(f: => Any) {
    val thrown = intercept[E](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    val msg =
      s"""expected fatal exception with pattern `$regex'
         |  Found: ${ thrown.getMessage } """
    if (!p)
      println(msg)
    assert(p, msg)
  }
  def interceptFatal(regex: String)(f: => Any) {
    interceptException[HailException](regex)(f)
  }

  def interceptSpark(regex: String)(f: => Any) {
    interceptException[SparkException](regex)(f)
  }

  def interceptAssertion(regex: String)(f: => Any) {
    interceptException[AssertionError](regex)(f)
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
      .annotateRowsExpr("AC" -> "AGG.map(g => g.gt.toInt64()).sum().toInt32()",
                        "ACsq" -> "AGG.map(g => (g.gt * g.gt).toInt64()).sum().toInt32()",
                        "nCalled" -> "AGG.filter(g => isDefined(g.gt)).count().toInt32()")
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
    mt.selectCols("""{fam_id: "0", id: sa.s, mat_id: "0", pat_id: "0", is_female: "0", pheno: "NA"}""", Some(FastIndexedSeq()))
      .annotateRowsExpr(
        "varid" -> """let l = va.locus and a = va.alleles in [l.contig, str(l.position), a[0], a[1]].mkString(":")""",
        "cm_position" -> "0.0")
      .exportPlink(path)
  }

  def eval(x: IR): Any = eval(x, Env.empty, FastIndexedSeq(), None)

  def eval(x: IR, agg: (IndexedSeq[Row], TStruct)): Any = eval(x, Env.empty, FastIndexedSeq(), Some(agg))

  def eval(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)]): Any = {
    val inputTypesB = new ArrayBuilder[Type]()
    val inputsB = new ArrayBuilder[Any]()

    args.foreach { case (v, t) =>
      inputsB += v
      inputTypesB += t
    }

    env.m.foreach { case (name, (v, t)) =>
      inputsB += v
      inputTypesB += t
    }

    val argsType = TTuple(inputTypesB.result(): _*)
    val resultType = TTuple(x.typ)
    val argsVar = genUID()

    val (_, substEnv) = env.m.foldLeft((args.length, Env.empty[IR])) { case ((i, env), (name, (v, t))) =>
      (i + 1, env.bind(name, GetTupleElement(Ref(argsVar, argsType), i)))
    }

    def rewrite(x: IR): IR = {
      x match {
        case In(i, t) =>
          GetTupleElement(Ref(argsVar, argsType), i)
        case _ =>
          Recur(rewrite)(x)
      }
    }

    agg match {
      case Some((aggElements, aggType)) =>
        val aggVar = genUID()
        val substAggEnv = aggType.fields.foldLeft(Env.empty[IR]) { case (env, f) =>
            env.bind(f.name, GetField(Ref(aggVar, aggType), f.name))
        }
        val (rvAggs, initOps, seqOps, aggResultType, postAggIR) = CompileWithAggregators[Long, Long, Long](
          argsVar, argsType,
          argsVar, argsType,
          aggVar, aggType,
          MakeTuple(FastSeq(rewrite(Subst(x, substEnv, substAggEnv)))), "AGGR",
          (i, x) => x,
          (i, x) => x)

        val (resultType2, f) = Compile[Long, Long, Long](
          "AGGR", aggResultType,
          argsVar, argsType,
          postAggIR)
        assert(resultType2 == resultType)

        Region.scoped { region =>
          val rvb = new RegionValueBuilder(region)

          // copy args into region
          rvb.start(argsType)
          rvb.startTuple()
          var i = 0
          while (i < inputsB.length) {
            rvb.addAnnotation(inputTypesB(i), inputsB(i))
            i += 1
          }
          rvb.endTuple()
          val argsOff = rvb.end()

          // aggregate
          i = 0
          rvAggs.foreach(_.clear())
          initOps()(region, rvAggs, argsOff, false)
          while (i < (aggElements.length / 2)) {
            // FIXME use second region for elements
            rvb.start(aggType)
            rvb.addAnnotation(aggType, aggElements(i))
            val aggElementOff = rvb.end()

            seqOps()(region, rvAggs, argsOff, false, aggElementOff, false)

            i += 1
          }

          val rvAggs2 = rvAggs.map(_.newInstance())
          rvAggs2.foreach(_.clear())
          initOps()(region, rvAggs2, argsOff, false)
          while (i < aggElements.length) {
            // FIXME use second region for elements
            rvb.start(aggType)
            rvb.addAnnotation(aggType, aggElements(i))
            val aggElementOff = rvb.end()

            seqOps()(region, rvAggs2, argsOff, false, aggElementOff, false)

            i += 1
          }

          rvAggs.zip(rvAggs2).foreach{ case(agg1, agg2) => agg1.combOp(agg2) }

          // build aggregation result
          rvb.start(aggResultType)
          rvb.startStruct()
          i = 0
          while (i < rvAggs.length) {
            rvAggs(i).result(rvb)
            i += 1
          }
          rvb.endStruct()
          val aggResultsOff = rvb.end()

          val resultOff = f()(region, aggResultsOff, false, argsOff, false)
          SafeRow(resultType.asInstanceOf[TBaseStruct], region, resultOff).get(0)
        }

      case None =>
        val (resultType2, f) = Compile[Long, Long](
          argsVar, argsType,
          MakeTuple(FastSeq(rewrite(Subst(x, substEnv)))))
        assert(resultType2 == resultType)

        Region.scoped { region =>
          val rvb = new RegionValueBuilder(region)
          rvb.start(argsType)
          rvb.startTuple()
          var i = 0
          while (i < inputsB.length) {
            rvb.addAnnotation(inputTypesB(i), inputsB(i))
            i += 1
          }
          rvb.endTuple()
          val argsOff = rvb.end()

          val resultOff = f()(region, argsOff, false)
          SafeRow(resultType.asInstanceOf[TBaseStruct], region, resultOff).get(0)
        }
    }
  }

  def assertEvalSame(x: IR) {
    assertEvalSame(x, Env.empty, FastIndexedSeq(), None)
  }

  def assertEvalSame(x: IR, agg: (IndexedSeq[Row], TStruct)) {
    assertEvalSame(x, Env.empty, FastIndexedSeq(), Some(agg))
  }

  def assertEvalSame(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)]) {
    val t = x.typ

    val i = Interpret[Any](x, env, args, agg)
    val i2 = Interpret[Any](x, env, args, agg, optimize = false)
    val c = eval(x, env, args, agg)

    assert(t.typeCheck(i))
    assert(t.typeCheck(i2))
    assert(t.typeCheck(c))

    assert(t.valuesSimilar(i, c))
    assert(t.valuesSimilar(i2, c))
  }

  def assertEvalsTo(x: IR, expected: Any) {
    assertEvalsTo(x, Env.empty, FastIndexedSeq(), None, expected)
  }

  def assertEvalsTo(x: IR, args: IndexedSeq[(Any, Type)], expected: Any) {
    assertEvalsTo(x, Env.empty, args, None, expected)
  }

  def assertEvalsTo(x: IR, agg: (IndexedSeq[Row], TStruct), expected: Any) {
    assertEvalsTo(x, Env.empty, FastIndexedSeq(), Some(agg), expected)
  }

  def assertEvalsTo(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)], expected: Any) {
    val t = x.typ
    assert(t.typeCheck(expected))

    val i = Interpret[Any](x, env, args, agg)
    assert(t.typeCheck(i))
    assert(t.valuesSimilar(i, expected), s"$i, $expected")

    val i2 = Interpret[Any](x, env, args, agg, optimize = false)
    assert(t.typeCheck(i2))
    assert(t.valuesSimilar(i2, expected), s"$i2 $expected")

    if (Compilable(x)) {
      val c = eval(x, env, args, agg)
      assert(t.typeCheck(c))
      assert(t.valuesSimilar(c, expected), s"$c, $expected")
    }
  }

  def assertThrows[E <: Throwable : Manifest](x: IR, regex: String) {
    assertThrows[E](x, Env.empty[(Any, Type)], FastIndexedSeq.empty[(Any, Type)], None, regex)
  }

  def assertThrows[E <: Throwable : Manifest](x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)], regex: String) {
    interceptException[E](regex)(Interpret[Any](x, env, args, agg))
    interceptException[E](regex)(Interpret[Any](x, env, args, agg, optimize = false))
    interceptException[E](regex)(eval(x, env, args, agg))
  }

  def assertFatal(x: IR, regex: String) {
    assertThrows[HailException](x, regex)
  }

  def assertFatal(x: IR, args: IndexedSeq[(Any, Type)], regex: String) {
    assertThrows[HailException](x, Env.empty[(Any, Type)], args, None, regex)
  }

  def assertFatal(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)], regex: String) {
    assertThrows[HailException](x, env, args, agg, regex)
  }

  def assertCompiledThrows[E <: Throwable : Manifest](x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)], regex: String) {
    interceptException[E](regex)(eval(x, env, args, agg))
  }

  def assertCompiledThrows[E <: Throwable : Manifest](x: IR, regex: String) {
    assertCompiledThrows[E](x, Env.empty[(Any, Type)], FastIndexedSeq.empty[(Any, Type)], None, regex)
  }

  def assertCompiledFatal(x: IR, regex: String) {
    assertCompiledThrows[HailException](x, regex)
  }
}
