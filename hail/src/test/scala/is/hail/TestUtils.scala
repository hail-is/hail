package is.hail

import java.io.{File, PrintWriter}

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir._
import is.hail.expr.ir.{BindingEnv, MakeTuple, Subst}
import is.hail.expr.ir.lowering.LowererUnsupportedOperation
import is.hail.expr.types.MatrixType
import is.hail.expr.types.physical.{PArray, PBaseStruct, PCanonicalString, PCanonicalTuple, PStruct, PTuple, PTupleField, PType}
import is.hail.expr.types.virtual._
import is.hail.io.plink.MatrixPLINKReader
import is.hail.io.vcf.MatrixVCFReader
import is.hail.utils.{ExecutionTimer, _}
import is.hail.variant._
import org.apache.spark.SparkException
import org.apache.spark.sql.Row

object ExecStrategy extends Enumeration {
  type ExecStrategy = Value
  val Interpret, InterpretUnoptimized, JvmCompile, LoweredJVMCompile, JvmCompileUnoptimized = Value

  val compileOnly: Set[ExecStrategy] = Set(JvmCompile, JvmCompileUnoptimized)
  val javaOnly: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized, JvmCompile, JvmCompileUnoptimized)
  val interpretOnly: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized)
  val nonLowering: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized, JvmCompile, JvmCompileUnoptimized)
  val lowering: Set[ExecStrategy] = Set(LoweredJVMCompile)
  val backendOnly: Set[ExecStrategy] = Set(LoweredJVMCompile)
  val allRelational: Set[ExecStrategy] = interpretOnly.union(lowering)
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

  def loweredExecute(x: IR, env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    bytecodePrinter: Option[PrintWriter] = None
  ): Any = {
    if (agg.isDefined || !env.isEmpty || !args.isEmpty)
      throw new LowererUnsupportedOperation("can't test with aggs or user defined args/env")
    HailContext.backend.jvmLowerAndExecute(x, optimize = false, lowerTable = true, lowerBM = true, print = bytecodePrinter)._1
  }

  def eval(x: IR): Any = eval(x, Env.empty, FastIndexedSeq(), None)

  def eval(x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    bytecodePrinter: Option[PrintWriter] = None,
    optimize: Boolean = true
  ): Any = {
    ExecuteContext.scoped { ctx =>
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

      val argsPType = PType.canonical(argsType)
      agg match {
        case Some((aggElements, aggType)) =>
          val aggElementVar = genUID()
          val aggArrayVar = genUID()
          val aggPType = PType.canonical(aggType)
          val aggArrayPType = PArray(aggPType)

          val substAggEnv = aggType.fields.foldLeft(Env.empty[IR]) { case (env, f) =>
            env.bind(f.name, GetField(Ref(aggElementVar, aggType), f.name))
          }
          val aggIR = StreamAgg(ToStream(Ref(aggArrayVar, aggArrayPType.virtualType)),
            aggElementVar,
            MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(eval = substEnv, agg = Some(substAggEnv)))))))

          val (resultType2, f) = Compile[Long, Long, Long](ctx,
            argsVar, argsPType,
            aggArrayVar, aggArrayPType,
            aggIR,
            print = bytecodePrinter,
            optimize = optimize)
          assert(resultType2.virtualType == resultType)

          Region.scoped { region =>
            val rvb = new RegionValueBuilder(region)
            rvb.start(argsPType)
            rvb.startTuple()
            var i = 0
            while (i < inputsB.length) {
              rvb.addAnnotation(inputTypesB(i), inputsB(i))
              i += 1
            }
            rvb.endTuple()
            val argsOff = rvb.end()

            rvb.start(aggArrayPType)
            rvb.startArray(aggElements.length)
            aggElements.foreach { r =>
              rvb.addAnnotation(aggType, r)
            }
            rvb.endArray()
            val aggOff = rvb.end()

            val resultOff = f(0, region)(region, argsOff, false, aggOff, false)
            SafeRow(resultType2.asInstanceOf[PBaseStruct], resultOff).get(0)
          }

        case None =>
          val (resultType2, f) = Compile[Long, Long](ctx,
            argsVar, argsPType,
            MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(substEnv))))),
            optimize = optimize,
            print = bytecodePrinter)
          assert(resultType2.virtualType == resultType)

          Region.scoped { region =>
            val rvb = new RegionValueBuilder(region)
            rvb.start(argsPType)
            rvb.startTuple()
            var i = 0
            while (i < inputsB.length) {
              rvb.addAnnotation(inputTypesB(i), inputsB(i))
              i += 1
            }
            rvb.endTuple()
            val argsOff = rvb.end()

            val resultOff = f(0, region)(region, argsOff, false)
            SafeRow(resultType2.asInstanceOf[PBaseStruct], resultOff).get(0)
          }
      }
    }
  }

  def assertEvalSame(x: IR) {
    assertEvalSame(x, Env.empty, FastIndexedSeq())
  }

  def assertEvalSame(x: IR, args: IndexedSeq[(Any, Type)]) {
    assertEvalSame(x, Env.empty, args)
  }

  def assertEvalSame(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)]) {
    val t = x.typ

    val (i, i2, c) = ExecuteContext.scoped { ctx =>
      val i = Interpret[Any](ctx, x, env, args)
      val i2 = Interpret[Any](ctx, x, env, args, optimize = false)
      val c = eval(x, env, args, None)
      (i, i2, c)
    }

    assert(t.typeCheck(i))
    assert(t.typeCheck(i2))
    assert(t.typeCheck(c))

    assert(t.valuesSimilar(i, c), s"interpret $i vs compile $c")
    assert(t.valuesSimilar(i2, c), s"interpret (optimize = false) $i vs compile $c")
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
    assert(t.typeCheck(expected), s"$t, $expected")

    ExecuteContext.scoped { ctx =>
      val filteredExecStrats: Set[ExecStrategy] =
        if (HailContext.backend.isInstanceOf[SparkBackend]) execStrats
        else {
          info("skipping interpret and non-lowering compile steps on non-spark backend")
          execStrats.intersect(ExecStrategy.backendOnly)
        }

      filteredExecStrats.foreach { strat =>
        InferPType.clearPTypes(x)
        try {
          val res = strat match {
            case ExecStrategy.Interpret =>
              assert(agg.isEmpty)
              Interpret[Any](ctx, x, env, args)
            case ExecStrategy.InterpretUnoptimized =>
              assert(agg.isEmpty)
              Interpret[Any](ctx, x, env, args, optimize = false)
            case ExecStrategy.JvmCompile =>
              assert(Forall(x, node => Compilable(node)))
              eval(x, env, args, agg, bytecodePrinter =
                Option(HailContext.getFlag("jvm_bytecode_dump"))
                  .map { path =>
                    val pw = new PrintWriter(new File(path))
                    pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(x)}\n */\n\n")
                    pw
                  })
            case ExecStrategy.JvmCompileUnoptimized =>
              assert(Forall(x, node => Compilable(node)))
              eval(x, env, args, agg, bytecodePrinter =
                Option(HailContext.getFlag("jvm_bytecode_dump"))
                  .map { path =>
                    val pw = new PrintWriter(new File(path))
                    pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(x)}\n */\n\n")
                    pw
                  },
                optimize = false)
            case ExecStrategy.LoweredJVMCompile =>
              loweredExecute(x, env, args, agg)
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
    assertThrows[E](x, Env.empty[(Any, Type)], FastIndexedSeq.empty[(Any, Type)], regex)
  }

  def assertThrows[E <: Throwable : Manifest](x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], regex: String) {
    ExecuteContext.scoped { ctx =>
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args))
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args, optimize = false))
      interceptException[E](regex)(eval(x, env, args, None))
    }
  }

  def assertFatal(x: IR, regex: String) {
    assertThrows[HailException](x, regex)
  }

  def assertFatal(x: IR, args: IndexedSeq[(Any, Type)], regex: String) {
    assertThrows[HailException](x, Env.empty[(Any, Type)], args, regex)
  }

  def assertFatal(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], regex: String) {
    assertThrows[HailException](x, env, args, regex)
  }

  def assertCompiledThrows[E <: Throwable : Manifest](x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], regex: String) {
    interceptException[E](regex)(eval(x, env, args, None))
  }

  def assertCompiledThrows[E <: Throwable : Manifest](x: IR, regex: String) {
    assertCompiledThrows[E](x, Env.empty[(Any, Type)], FastIndexedSeq.empty[(Any, Type)], regex)
  }

  def assertCompiledFatal(x: IR, regex: String) {
    assertCompiledThrows[HailException](x, regex)
  }

  def assertNDEvals(nd: NDArrayIR, expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {
    assertNDEvals(nd, Env.empty, FastIndexedSeq(), None, expected)
  }

  def assertNDEvals(nd: NDArrayIR, expected: (Any, IndexedSeq[Long]))
    (implicit execStrats: Set[ExecStrategy]) {
    if (expected == null)
      assertNDEvals(nd, Env.empty, FastIndexedSeq(), None, null, null)
    else
      assertNDEvals(nd, Env.empty, FastIndexedSeq(), None, expected._2, expected._1)
  }

  def assertNDEvals(nd: NDArrayIR, args: IndexedSeq[(Any, Type)], expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {
    assertNDEvals(nd, Env.empty, args, None, expected)
  }

  def assertNDEvals(nd: NDArrayIR, agg: (IndexedSeq[Row], TStruct), expected: Any)
    (implicit execStrats: Set[ExecStrategy]) {
    assertNDEvals(nd, Env.empty, FastIndexedSeq(), Some(agg), expected)
  }

  def assertNDEvals(nd: NDArrayIR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)], expected: Any)
    (implicit execStrats: Set[ExecStrategy]): Unit = {
    var e: IndexedSeq[Any] = expected.asInstanceOf[IndexedSeq[Any]]
    val dims = Array.fill(nd.typ.nDims) {
      val n = e.length
      if (n != 0 && e.head.isInstanceOf[IndexedSeq[_]])
        e = e.head.asInstanceOf[IndexedSeq[Any]]
      n.toLong
    }
    assertNDEvals(nd, Env.empty, FastIndexedSeq(), agg, dims, expected)
  }

  def assertNDEvals(nd: NDArrayIR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)], dims: IndexedSeq[Long], expected: Any)
    (implicit execStrats: Set[ExecStrategy]): Unit = {
    val arrayIR = if (expected == null) nd else {
      val refs = Array.fill(nd.typ.nDims) { Ref(genUID(), TInt32) }
      Let("nd", nd,
        dims.zip(refs).foldRight[IR](NDArrayRef(Ref("nd", nd.typ), refs.map(Cast(_, TInt64)))) {
          case ((n, ref), accum) =>
            ToArray(StreamMap(rangeIR(n.toInt), ref.name, accum))
        })
    }
    assertEvalsTo(arrayIR, env, args, agg, expected)
  }

  def assertBMEvalsTo(bm: BlockMatrixIR, expected: DenseMatrix[Double])
    (implicit execStrats: Set[ExecStrategy]): Unit = {
    ExecuteContext.scoped { ctx =>
      val filteredExecStrats: Set[ExecStrategy] =
        if (HailContext.backend.isInstanceOf[SparkBackend]) execStrats
        else {
          info("skipping interpret and non-lowering compile steps on non-spark backend")
          execStrats.intersect(ExecStrategy.backendOnly)
        }
      filteredExecStrats.filter(ExecStrategy.interpretOnly).foreach { strat =>
        try {
          val res = strat match {
            case ExecStrategy.Interpret =>
              Interpret(bm, ctx, optimize = true)
            case ExecStrategy.InterpretUnoptimized =>
              Interpret(bm, ctx, optimize = false)
          }
          assert(res.toBreezeMatrix() == expected)
        } catch {
          case e: Exception =>
            error(s"error from strategy $strat")
            if (execStrats.contains(strat)) throw e
        }
      }
      val expectedArray = Array.tabulate(expected.rows)(i => Array.tabulate(expected.cols)(j => expected(i, j)).toFastIndexedSeq).toFastIndexedSeq
      assertNDEvals(BlockMatrixCollect(bm), expectedArray)(filteredExecStrats.filterNot(ExecStrategy.interpretOnly))
    }
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
    partitionsJSON: String = null): MatrixIR = {
    rg.foreach { referenceGenome =>
      ReferenceGenome.addReference(referenceGenome)
    }
    val entryFloatType = TFloat64._toPretty

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
    MatrixRead(reader.fullMatrixType, dropSamples, false, reader)
  }
}
