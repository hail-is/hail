package is.hail

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.backend.{LowerTableIR, LowererUnsupportedOperation}
import is.hail.backend.spark.SparkBackend
import is.hail.cxx.CXXUnsupportedOperation
import is.hail.expr.ir._
import is.hail.expr.types.MatrixType
import is.hail.expr.types.virtual._
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.utils.{ExecutionTimer, _}
import is.hail.variant._
import org.apache.spark.SparkException
import org.apache.spark.sql.Row

object ExecStrategy extends Enumeration {
  type ExecStrategy = Value
  val Interpret, InterpretUnoptimized, JvmCompile, CxxCompile, LoweredJVMCompile = Value

  val javaOnly:Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized, JvmCompile)
  val interpretOnly: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized)
  val nonLowering: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized, JvmCompile, CxxCompile)
  val backendOnly: Set[ExecStrategy] = Set(LoweredJVMCompile, CxxCompile)
}

object TestUtils {

  import org.scalatest.Assertions._

  def interceptException[E <: Throwable : Manifest](regex: String)(f: => Any) {
    val thrown = intercept[E](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    val msg =
      s"""expected fatal exception with pattern '$regex'
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

  def unphasedDiploidGtIndicesToBoxedCall(m: DenseMatrix[Int]): DenseMatrix[BoxedCall] = {
    m.map(g => if (g == -1) null: BoxedCall else Call2.fromUnphasedDiploidGtIndex(g): BoxedCall)
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

  def nativeExecute(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)]): Any = {
    if (agg.isDefined)
      throw new CXXUnsupportedOperation

    if (env.m.isEmpty && args.isEmpty) {
      try {
        val (res, _) = HailContext.backend.cxxLowerAndExecute(x, optimize = false)
        res
      } catch {
        case e: CXXUnsupportedOperation =>
          throw e
      }
    } else {
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
        (i + 1, env.bind(name, GetTupleElement(In(0, argsType), i)))
      }

      def rewrite(x: IR): IR = {
        x match {
          case In(i, t) =>
            GetTupleElement(In(0, argsType), i)
          case _ =>
            MapIR(rewrite)(x)
        }
      }

      val rewritten = Subst(rewrite(x), BindingEnv(substEnv))
      val f = cxx.Compile(
        argsVar, argsType.physicalType,
        MakeTuple.ordered(FastSeq(rewritten)), false)

      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        rvb.start(argsType.physicalType)
        rvb.startTuple()
        var i = 0
        while (i < inputsB.length) {
          rvb.addAnnotation(inputTypesB(i), inputsB(i))
          i += 1
        }
        rvb.endTuple()
        val argsOff = rvb.end()

        val resultOff = f(region.get(), argsOff)
        SafeRow(resultType.asInstanceOf[TBaseStruct].physicalType, region, resultOff).get(0)
      }
    }
  }

  def loweredExecute(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)]): Any = {
    if (agg.isDefined || !env.isEmpty || !args.isEmpty)
      throw new LowererUnsupportedOperation("can't test with aggs or user defined args/env")
    HailContext.backend.jvmLowerAndExecute(x, optimize = false)._1
  }

  def eval(x: IR): Any = eval(x, Env.empty, FastIndexedSeq(), None)

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
          MapIR(rewrite)(x)
      }
    }

    agg match {
      case Some((aggElements, aggType)) =>
        val aggVar = genUID()
        val substAggEnv = aggType.fields.foldLeft(Env.empty[IR]) { case (env, f) =>
            env.bind(f.name, GetField(Ref(aggVar, aggType), f.name))
        }
        val (rvAggs, initOps, seqOps, aggResultType, postAggIR) = CompileWithAggregators[Long, Long, Long](
          argsVar, argsType.physicalType,
          argsVar, argsType.physicalType,
          aggVar, aggType.physicalType,
          MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(eval = substEnv, agg = Some(substAggEnv)))))), "AGGR",
          (i, x) => x,
          (i, x) => x)

        val (resultType2, f) = Compile[Long, Long, Long](
          "AGGR", aggResultType,
          argsVar, argsType.physicalType,
          postAggIR)
        assert(resultType2.virtualType == resultType)

        Region.scoped { region =>
          val rvb = new RegionValueBuilder(region)

          // copy args into region
          rvb.start(argsType.physicalType)
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
          initOps(0, region)(region, rvAggs, argsOff, false)
          var seqOpF = seqOps(0, region)
          while (i < (aggElements.length / 2)) {
            // FIXME use second region for elements
            rvb.start(aggType.physicalType)
            rvb.addAnnotation(aggType, aggElements(i))
            val aggElementOff = rvb.end()

            seqOpF(region, rvAggs, argsOff, false, aggElementOff, false)

            i += 1
          }

          val rvAggs2 = rvAggs.map(_.newInstance())
          rvAggs2.foreach(_.clear())
          initOps(0, region)(region, rvAggs2, argsOff, false)
          seqOpF = seqOps(1, region)
          while (i < aggElements.length) {
            // FIXME use second region for elements
            rvb.start(aggType.physicalType)
            rvb.addAnnotation(aggType, aggElements(i))
            val aggElementOff = rvb.end()

            seqOpF(region, rvAggs2, argsOff, false, aggElementOff, false)

            i += 1
          }

          rvAggs.zip(rvAggs2).foreach{ case(agg1, agg2) => agg1.combOp(agg2) }

          // build aggregation result
          rvb.start(aggResultType)
          rvb.startTuple()
          i = 0
          while (i < rvAggs.length) {
            rvAggs(i).result(rvb)
            i += 1
          }
          rvb.endTuple()
          val aggResultsOff = rvb.end()

          val resultOff = f(0, region)(region, aggResultsOff, false, argsOff, false)
          SafeRow(resultType.asInstanceOf[TBaseStruct].physicalType, region, resultOff).get(0)
        }

      case None =>
        val (resultType2, f) = Compile[Long, Long](
          argsVar, argsType.physicalType,
          MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(substEnv))))))
        assert(resultType2.virtualType == resultType)

        Region.scoped { region =>
          val rvb = new RegionValueBuilder(region)
          rvb.start(argsType.physicalType)
          rvb.startTuple()
          var i = 0
          while (i < inputsB.length) {
            rvb.addAnnotation(inputTypesB(i), inputsB(i))
            i += 1
          }
          rvb.endTuple()
          val argsOff = rvb.end()

          val resultOff = f(0, region)(region, argsOff, false)
          SafeRow(resultType.asInstanceOf[TBaseStruct].physicalType, region, resultOff).get(0)
        }
    }
  }

  def assertEvalSame(x: IR) {
    assertEvalSame(x, Env.empty, FastIndexedSeq(), None)
  }

  def assertEvalSame(x: IR, args: IndexedSeq[(Any, Type)]) {
    assertEvalSame(x, Env.empty, args, None)
  }

  def assertEvalSame(x: IR, agg: (IndexedSeq[Row], TStruct)) {
    assertEvalSame(x, Env.empty, FastIndexedSeq(), Some(agg))
  }

  def assertEvalSame(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)]) {
    val t = x.typ

    val (i, i2, c) = ExecuteContext.scoped { ctx =>
      val i = Interpret[Any](ctx, x, env, args, agg)
      val i2 = Interpret[Any](ctx, x, env, args, agg, optimize = false)
      val c = eval(x, env, args, agg)
      (i, i2, c)
    }

    assert(t.typeCheck(i))
    assert(t.typeCheck(i2))
    assert(t.typeCheck(c))

    assert(t.valuesSimilar(i, c), s"interpret $i vs compile $c")
    assert(t.valuesSimilar(i2, c), s"interpret (optimize = false) $i vs compile $c")

    try {
      val c2 = nativeExecute(x, env, args, agg)
      assert(t.typeCheck(c2))
      assert(t.valuesSimilar(c2, c), s"native compile $c2 vs compile $c")
    } catch {
      case _: CXXUnsupportedOperation =>
    }
  }

  def assertAllEvalTo(xs: (IR, Any)*)(implicit execStrats: Set[ExecStrategy]): Unit = {
    assertEvalsTo(MakeTuple.ordered(xs.map(_._1)), Row.fromSeq(xs.map(_._2)))
  }


  def assertEvalsTo(x: IR, expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {
    assertEvalsTo(x, Env.empty, FastIndexedSeq(), None, expected)
  }

  def assertEvalsTo(x: IR, args: IndexedSeq[(Any, Type)], expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {
    assertEvalsTo(x, Env.empty, args, None, expected)
  }

  def assertEvalsTo(x: IR, agg: (IndexedSeq[Row], TStruct), expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {
    assertEvalsTo(x, Env.empty, FastIndexedSeq(), Some(agg), expected)
  }

  def assertEvalsTo(x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {

    TypeCheck(x, BindingEnv(env.mapValues(_._2), agg = agg.map(_._2.toEnv)))

    val t = x.typ
    assert(t.typeCheck(expected), t)

    ExecuteContext.scoped { ctx =>
      val filteredExecStrats: Set[ExecStrategy] =
        if (HailContext.backend.isInstanceOf[SparkBackend]) execStrats
        else {
          info("skipping interpret and non-lowering compile steps on non-spark backend")
          execStrats.intersect(ExecStrategy.backendOnly)
        }

      filteredExecStrats.foreach { strat =>
        try {
          val res = strat match {
            case ExecStrategy.Interpret => Interpret[Any](ctx, x, env, args, agg)
            case ExecStrategy.InterpretUnoptimized => Interpret[Any](ctx, x, env, args, agg, optimize = false)
            case ExecStrategy.JvmCompile =>
              assert(Forall(x, node => node.isInstanceOf[IR] && Compilable(node.asInstanceOf[IR])))
              eval(x, env, args, agg)
            case ExecStrategy.CxxCompile => nativeExecute(x, env, args, agg)
            case ExecStrategy.LoweredJVMCompile => loweredExecute(x, env, args, agg)
          }
          assert(t.typeCheck(res))
          assert(t.valuesSimilar(res, expected), s"\n  result=$res\n  expect=$expected\n  strategy=$strat)")
        } catch {
          case e: Exception =>
            error(s"error from strategy $strat")
            if (execStrats.contains(strat)) throw e
        }
      }
    }
  }

  def assertThrows[E <: Throwable : Manifest](x: IR, regex: String) {
    assertThrows[E](x, Env.empty[(Any, Type)], FastIndexedSeq.empty[(Any, Type)], None, regex)
  }

  def assertThrows[E <: Throwable : Manifest](x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], agg: Option[(IndexedSeq[Row], TStruct)], regex: String) {
    ExecuteContext.scoped { ctx =>
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args, agg))
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args, agg, optimize = false))
      interceptException[E](regex)(eval(x, env, args, agg))
    }
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


  def importVCF(hc: HailContext, file: String, force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String],
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.GRCh37),
    contigRecoding: Option[Map[String, String]] = None,
    arrayElementsRequired: Boolean = true,
    skipInvalidLoci: Boolean = false,
    partitionsJSON: String = null): MatrixTable = {
    rg.foreach { referenceGenome =>
      ReferenceGenome.addReference(referenceGenome)
    }
    val entryFloatType = TFloat64()._toPretty

    val reader = MatrixVCFReader(
      Array(file),
      callFields,
      entryFloatType,
      headerFile,
      nPartitions,
      rg.map(_.name),
      contigRecoding.getOrElse(Map.empty[String, String]),
      arrayElementsRequired,
      skipInvalidLoci,
      forceBGZ,
      force,
      TextInputFilterAndReplace(),
      partitionsJSON
    )
    new MatrixTable(hc, MatrixRead(reader.fullMatrixType, dropSamples, false, reader))
  }

  def vdsFromCallMatrix(hc: HailContext)(
    callMat: Matrix[BoxedCall],
    samplesIdsOpt: Option[Array[String]] = None,
    nPartitions: Int = hc.sc.defaultMinPartitions): MatrixTable = {

    require(samplesIdsOpt.forall(_.length == callMat.rows))
    require(samplesIdsOpt.forall(_.areDistinct()))

    val sampleIds = samplesIdsOpt.getOrElse((0 until callMat.rows).map(_.toString).toArray)

    val rdd = hc.sc.parallelize(
      (0 until callMat.cols).map { j =>
        (Annotation(Locus("1", j + 1), FastIndexedSeq("A", "C")),
          (0 until callMat.rows).map { i =>
            Genotype(callMat(i, j))
          }: Iterable[Annotation])
      },
      nPartitions)

    MatrixTable.fromLegacy(hc, MatrixType(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = TStruct("s" -> TString()),
      rowKey = Array("locus", "alleles"),
      rowType = TStruct("locus" -> TLocus(ReferenceGenome.GRCh37),
        "alleles" -> TArray(TString())),
      entryType = Genotype.htsGenotypeType),
      Annotation.empty, sampleIds.map(Annotation(_)), rdd)
  }
}
