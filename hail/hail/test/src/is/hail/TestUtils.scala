package is.hail

import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations.{Region, RegionPool, RegionValueBuilder, SafeRow}
import is.hail.asm4s.{
  classInfo, AsmFunction2RegionLongLong, AsmFunction3RegionLongLongLong, LongInfo,
}
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkBackend
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir.{
  bindIR, freshName, rangeIR, streamAggIR, BindingEnv, BlockMatrixIR, Compilable, Compile, Env,
  Forall, IR, Interpret, MapIR, MatrixIR, MatrixRead, Name, Pretty, SingleCodeEmitParamType, Subst,
  TypeCheck,
}
import is.hail.expr.ir.Optimize.Flags.Optimize
import is.hail.expr.ir.defs.{
  BlockMatrixCollect, Cast, GetField, GetTupleElement, In, MakeTuple, NDArrayRef, Ref, StreamMap,
  ToArray, ToStream,
}
import is.hail.expr.ir.lowering.LowererUnsupportedOperation
import is.hail.io.vcf.MatrixVCFReader
import is.hail.types.physical.{PBaseStruct, PCanonicalArray, PType}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.{BoxedCall, Call2, ReferenceGenome}

import scala.reflect.ClassTag

import java.io.PrintWriter

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.junit.jupiter.api.{Assertions => JAssertions}
import org.junit.jupiter.api.Assertions.assertTrue
import org.scalacheck.{Prop, Test => ScalaCheckTest}
import org.scalacheck.rng.Seed
import org.scalatest

object TestUtils extends Logging {

  def intercept[E <: Throwable: ClassTag](f: => Any): E =
    JAssertions.assertThrows(
      implicitly[ClassTag[E]].runtimeClass.asInstanceOf[Class[E]],
      () => { f; () },
    )

  def interceptException[E <: Throwable: ClassTag](regex: String)(f: => Any): Unit = {
    val thrown = intercept[E](f)
    val msg = thrown.getMessage
    val matched = msg != null && regex.r.findFirstIn(msg).isDefined
    assertTrue(
      matched,
      s"expected exception with pattern '$regex'\n  Found: $msg",
    )
  }

  def interceptFatal(regex: String)(f: => Any): Unit =
    interceptException[HailException](regex)(f)

  def interceptSpark(regex: String)(f: => Any): Unit =
    interceptException[SparkException](regex)(f)

  def interceptAssertion(regex: String)(f: => Any): Unit =
    interceptException[AssertionError](regex)(f)

  def cartesian[A, B](as: Iterable[A], bs: Iterable[B]): Iterable[(A, B)] =
    for {
      a <- as
      b <- bs
    } yield (a, b)

  /** Scala-conventional `(actual, expected)` wrapper around JUnit's `assertEquals(expected,
    * actual)`. Prefer this over calling `assertEquals` directly in migrated suites: the argument
    * order here matches the order the old scalatest / idiomatic Scala assertions used, avoiding
    * silent argument inversions that would swap the "expected" and "actual" labels on failure.
    */
  // scalafix:off ForbiddenSymbol
  def assertEq[A](actual: A, expected: A): Unit =
    JAssertions.assertEquals(expected, actual)

  def assertEq[A](actual: A, expected: A, message: String): Unit =
    JAssertions.assertEquals(expected, actual, message)
  // scalafix:on ForbiddenSymbol

  // --- ScalaCheck integration ---

  private val scalaCheckSeedOverride: Option[Long] =
    Option(System.getProperty("scalacheck.seed"))
      .orElse(Option(System.getenv("SCALACHECK_SEED")))
      .map(_.toLong)

  /** Runs a ScalaCheck [[Prop]] with a deterministic-but-overridable seed. Fails the enclosing
    * JUnit test (via an `AssertionError`) on anything other than `Passed`/`Proved` and prints the
    * seed so the run can be reproduced with `-Dscalacheck.seed=<n>` or `SCALACHECK_SEED=<n>`.
    */
  def check(prop: Prop): Unit = {
    val baseSeed = scalaCheckSeedOverride.getOrElse(System.nanoTime())
    val params = ScalaCheckTest.Parameters.default.withInitialSeed(Seed(baseSeed))
    val result = ScalaCheckTest.check(params, prop)
    result.status match {
      case ScalaCheckTest.Passed | _: ScalaCheckTest.Proved => ()
      case _ =>
        throw new AssertionError(
          s"Property failed (seed=$baseSeed, reproduce with -Dscalacheck.seed=$baseSeed): $result"
        )
    }
  }

  // Allows the body of a `forAll` to have type `Unit`, relying on assertions to trigger failure
  implicit def unitToProp(unit: Unit): Prop = Prop.proved
  implicit def assertionToProp(unit: scalatest.Assertion): Prop = Prop.proved

  // --- IR evaluation helpers (ported from TestUtils / HailSuite) ---

  def unoptimized[A](f: ExecuteContext => A)(implicit ctx: ExecuteContext): A =
    ctx.local(flags = ctx.flags - Optimize)(f)

  def loweredExecute(
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    bytecodePrinter: Option[PrintWriter] = None,
  )(implicit ctx: ExecuteContext
  ): Any = {
    if (agg.isDefined || !env.isEmpty || args.nonEmpty)
      throw new LowererUnsupportedOperation("can't test with aggs or user defined args/env")

    unoptimized { ctx =>
      ctx.backend.asSpark.jvmLowerAndExecute(
        ctx,
        x,
        lowerTable = true,
        lowerBM = true,
        print = bytecodePrinter,
      )
    }
  }

  def eval(
    x: IR,
    env: Env[(Any, Type)] = Env.empty,
    args: IndexedSeq[(Any, Type)] = FastSeq(),
    agg: Option[(IndexedSeq[Row], TStruct)] = None,
    bytecodePrinter: Option[PrintWriter] = None,
  )(implicit ctx: ExecuteContext
  ): Any = {
    val inputs = (args.view.map(_._1) ++ env.m.view.map(_._2._1)).to(ArraySeq)
    val inputTypes = (args.view.map(_._2) ++ env.m.view.map(_._2._2)).to(ArraySeq)

    val argsType = TTuple(inputTypes: _*)
    val resultType = TTuple(x.typ)
    val argsVar = Ref(freshName(), argsType)

    val (_, substEnv) =
      env.m.foldLeft((args.length, Env.empty[IR])) { case ((i, env), (name, (_, _))) =>
        (i + 1, env.bind(name, GetTupleElement(argsVar, i)))
      }

    def rewrite(x: IR): IR = x match {
      case In(i, _) => GetTupleElement(argsVar, i)
      case _ => MapIR(rewrite)(x)
    }

    val argsPType = PType.canonical(argsType).setRequired(true)
    agg match {
      case Some((aggElements, aggType)) =>
        val aggPType = PType.canonical(aggType)
        val aggArrayPType = PCanonicalArray(aggPType, required = true)
        val aggArrayVar = Ref(freshName(), aggArrayPType.virtualType)

        val aggIR = streamAggIR(ToStream(aggArrayVar)) { aggElementVar =>
          val substAggEnv = aggType.fields.foldLeft(Env.empty[IR]) { case (env, f) =>
            env.bind(Name(f.name), GetField(aggElementVar, f.name))
          }
          MakeTuple.ordered(FastSeq(rewrite(Subst(
            x,
            BindingEnv(eval = substEnv, agg = Some(substAggEnv)),
          ))))
        }

        val (Some(PTypeReferenceSingleCodeType(resultType2)), f) =
          Compile[AsmFunction3RegionLongLongLong](
            ctx,
            FastSeq(
              (
                argsVar.name,
                SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsPType)),
              ),
              (
                aggArrayVar.name,
                SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(aggArrayPType)),
              ),
            ),
            FastSeq(classInfo[Region], LongInfo, LongInfo),
            LongInfo,
            aggIR,
            print = bytecodePrinter,
          )
        assert(resultType2.virtualType == resultType)

        ctx.r.pool.scopedRegion { region =>
          ctx.local(r = region) { ctx =>
            val rvb = new RegionValueBuilder(ctx.stateManager, ctx.r)
            rvb.start(argsPType)
            rvb.startTuple()
            var i = 0
            while (i < inputs.length) {
              rvb.addAnnotation(inputTypes(i), inputs(i))
              i += 1
            }
            rvb.endTuple()
            val argsOff = rvb.end()

            rvb.start(aggArrayPType)
            rvb.startArray(aggElements.length)
            aggElements.foreach(r => rvb.addAnnotation(aggType, r))
            rvb.endArray()
            val aggOff = rvb.end()

            ctx.scopedExecution { (hcl, fs, tc, r) =>
              val off = f(hcl, fs, tc, r)(r, argsOff, aggOff)
              SafeRow(resultType2.asInstanceOf[PBaseStruct], off).get(0)
            }
          }
        }

      case None =>
        val (Some(PTypeReferenceSingleCodeType(resultType2)), f) =
          Compile[AsmFunction2RegionLongLong](
            ctx,
            FastSeq((
              argsVar.name,
              SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(argsPType)),
            )),
            FastSeq(classInfo[Region], LongInfo),
            LongInfo,
            MakeTuple.ordered(FastSeq(rewrite(Subst(x, BindingEnv(substEnv))))),
            print = bytecodePrinter,
          )
        assert(resultType2.virtualType == resultType)

        ctx.r.pool.scopedRegion { region =>
          ctx.local(r = region) { ctx =>
            val rvb = new RegionValueBuilder(ctx.stateManager, ctx.r)
            rvb.start(argsPType)
            rvb.startTuple()
            var i = 0
            while (i < inputs.length) {
              rvb.addAnnotation(inputTypes(i), inputs(i))
              i += 1
            }
            rvb.endTuple()
            val argsOff = rvb.end()
            ctx.scopedExecution { (hcl, fs, tc, r) =>
              val resultOff = f(hcl, fs, tc, r)(r, argsOff)
              SafeRow(resultType2.asInstanceOf[PBaseStruct], resultOff).get(0)
            }
          }
        }
    }
  }

  def evaluate(
    ir: IR,
    args: IndexedSeq[(Any, Type)],
    env: Env[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
  )(implicit
    ctx: ExecuteContext,
    strat: ExecStrategy,
  ): Any = strat match {
    case ExecStrategy.Interpret =>
      assert(agg.isEmpty)
      Interpret[Any](ctx, ir, env, args)
    case ExecStrategy.InterpretUnoptimized =>
      assert(agg.isEmpty)
      unoptimized(ctx => Interpret[Any](ctx, ir, env, args))
    case ExecStrategy.JvmCompile =>
      assert(Forall(ir, node => Compilable(node)))
      eval(
        ir,
        env,
        args,
        agg,
        bytecodePrinter = Option(ctx.getFlag("jvm_bytecode_dump")).map { path =>
          val pw = new PrintWriter(new java.io.File(path))
          pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, ir)}\n */\n\n")
          pw
        },
      )
    case ExecStrategy.JvmCompileUnoptimized =>
      assert(Forall(ir, node => Compilable(node)))
      unoptimized { implicit ctx =>
        eval(
          ir,
          env,
          args,
          agg,
          bytecodePrinter = Option(ctx.getFlag("jvm_bytecode_dump")).map { path =>
            val pw = new PrintWriter(new java.io.File(path))
            pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, ir)}\n */\n\n")
            pw
          },
        )
      }
    case ExecStrategy.LoweredJVMCompile =>
      loweredExecute(ir, env, args, agg)
  }

  def assertEvalsTo(
    x0: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit = {
    TypeCheck(ctx, x0, BindingEnv(env.mapValues(_._2), agg = agg.map(_._2.toEnv)))

    val t = x0.typ
    assert(t == TVoid || t.typeCheck(expected), s"$t, $expected")

    val filteredExecStrats: Set[ExecStrategy] =
      if (ctx.backend.isInstanceOf[SparkBackend])
        execStrats
      else {
        logger.info("skipping interpret and non-lowering compile steps on non-spark backend")
        execStrats.intersect(ExecStrategy.backendOnly)
      }

    val x = x0.deepCopy
    filteredExecStrats.foreach { implicit strat =>
      try {
        val res = evaluate(x, args, env, agg)

        if (t != TVoid) {
          assert(t.typeCheck(res), s"\n  t=$t\n  result=$res\n  strategy=$strat")
          assert(
            t.valuesSimilar(res, expected),
            s"\n  result=$res\n  expect=$expected\n  strategy=$strat)",
          )
        }
      } catch {
        case e: Exception =>
          logger.error(s"error from strategy $strat", e)
          if (execStrats.contains(strat)) throw e
      }
    }
  }

  def assertEvalsTo(
    x: IR,
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertEvalsTo(x, Env.empty, FastSeq(), None, expected)

  def assertEvalsTo(
    x: IR,
    args: IndexedSeq[(Any, Type)],
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertEvalsTo(x, Env.empty, args, None, expected)

  def assertEvalsTo(
    x: IR,
    agg: (IndexedSeq[Row], TStruct),
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertEvalsTo(x, Env.empty, FastSeq(), Some(agg), expected)

  private[this] lazy val testResources: String =
    sys.env.getOrElse("MILL_TEST_RESOURCE_DIR", "hail/test/resources")

  def getTestResource(localPath: String): String = s"$testResources/$localPath"

  def assertAllEvalTo(
    xs: (IR, Any)*
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertEvalsTo(
      MakeTuple.ordered(xs.toFastSeq.map(_._1)),
      Row.fromSeq(xs.map(_._2)),
    )

  def assertThrows[E <: Throwable: ClassTag](x: IR, regex: String)(implicit ctx: ExecuteContext)
    : Unit =
    ctx.local() { ctx =>
      val emptyEnv: Env[(Any, Type)] = Env.empty[(Any, Type)]
      val emptyArgs: IndexedSeq[(Any, Type)] = FastSeq.empty[(Any, Type)]
      ctx.flags.set(Optimize, "1")
      interceptException[E](regex)(Interpret[Any](ctx, x, emptyEnv, emptyArgs))
      ctx.flags.set(Optimize, null)
      interceptException[E](regex)(Interpret[Any](ctx, x, emptyEnv, emptyArgs))
      ctx.flags.set(Optimize, "1")
      interceptException[E](regex) {
        implicit val innerCtx: ExecuteContext = ctx
        eval(x)
      }
    }

  def assertFatal(x: IR, regex: String)(implicit ctx: ExecuteContext): Unit =
    assertThrows[HailException](x, regex)

  def assertThrows[E <: Throwable: ClassTag](
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    regex: String,
  )(implicit ctx: ExecuteContext
  ): Unit =
    ctx.local() { ctx =>
      ctx.flags.set(Optimize, "1")
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args))
      ctx.flags.set(Optimize, null)
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args))
      ctx.flags.set(Optimize, "1")
      interceptException[E](regex) {
        implicit val innerCtx: ExecuteContext = ctx
        eval(x, env, args)
      }
    }

  def assertFatal(
    x: IR,
    args: IndexedSeq[(Any, Type)],
    regex: String,
  )(implicit
    ctx: ExecuteContext
  ): Unit =
    assertThrows[HailException](x, Env.empty[(Any, Type)], args, regex)

  def assertFatal(
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    regex: String,
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertThrows[HailException](x, env, args, regex)

  def assertCompiledThrows[E <: Throwable: ClassTag](
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    regex: String,
  )(implicit ctx: ExecuteContext
  ): Unit =
    interceptException[E](regex)(eval(x, env, args))

  def assertCompiledThrows[E <: Throwable: ClassTag](
    x: IR,
    regex: String,
  )(implicit
    ctx: ExecuteContext
  ): Unit =
    assertCompiledThrows[E](x, Env.empty[(Any, Type)], FastSeq.empty[(Any, Type)], regex)

  def assertCompiledFatal(x: IR, regex: String)(implicit ctx: ExecuteContext): Unit =
    assertCompiledThrows[HailException](x, regex)

  def assertEvalSame(x: IR)(implicit ctx: ExecuteContext): Unit =
    assertEvalSame(x, Env.empty, FastSeq())

  def assertEvalSame(x: IR, args: IndexedSeq[(Any, Type)])(implicit ctx: ExecuteContext): Unit =
    assertEvalSame(x, Env.empty, args)

  def assertEvalSame(
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val t = x.typ
    val (i, i2, c) =
      ctx.local() { ctx =>
        ctx.flags.set(Optimize, "1")
        val i = Interpret[Any](ctx, x, env, args)
        ctx.flags.set(Optimize, null)
        val i2 = Interpret[Any](ctx, x, env, args)
        ctx.flags.set(Optimize, "1")
        val c = {
          implicit val innerCtx: ExecuteContext = ctx
          eval(x, env, args)
        }
        (i, i2, c)
      }
    assertTrue(t.typeCheck(i), s"interpret result $i does not type check")
    assertTrue(t.typeCheck(i2), s"interpret (unoptimized) result $i2 does not type check")
    assertTrue(t.typeCheck(c), s"compile result $c does not type check")
    assertTrue(t.valuesSimilar(i, c), s"interpret $i vs compile $c")
    assertTrue(t.valuesSimilar(i2, c), s"interpret (optimize = false) $i2 vs compile $c")
  }

  def importVCF(
    file: String,
    force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    blockSizeInMB: Option[Int] = None,
    minPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String],
    rg: Option[String] = Some(ReferenceGenome.GRCh37),
    contigRecoding: Option[Map[String, String]] = None,
    arrayElementsRequired: Boolean = true,
    skipInvalidLoci: Boolean = false,
    partitionsJSON: Option[String] = None,
    partitionsTypeStr: Option[String] = None,
  )(implicit ctx: ExecuteContext
  ): MatrixIR = {
    val entryFloatType = TFloat64._toPretty
    val reader = MatrixVCFReader(
      ctx,
      ArraySeq(file),
      callFields,
      entryFloatType,
      headerFile,
      /*sampleIDs=*/ None,
      nPartitions,
      blockSizeInMB,
      minPartitions,
      rg,
      contigRecoding.getOrElse(Map.empty[String, String]),
      arrayElementsRequired,
      skipInvalidLoci,
      forceBGZ,
      force,
      TextInputFilterAndReplace(),
      partitionsJSON,
      partitionsTypeStr,
    )
    MatrixRead(reader.fullMatrixTypeWithoutUIDs, dropSamples, false, reader)
  }

  def measuringHighestTotalMemoryUsage[A](
    f: ExecuteContext => A
  )(implicit
    ctx: ExecuteContext
  ): (A, Long) =
    RegionPool.scoped { p =>
      val a = ctx.local(r = Region(pool = p))(f)
      (a, p.getHighestTotalUsage)
    }

  def assertVectorEqualityDouble(
    A: Vector[Double],
    B: Vector[Double],
    tolerance: Double = defaultTolerance,
  ): Unit = {
    assertTrue(A.size == B.size, s"vector sizes differ: ${A.size} vs ${B.size}")
    assertTrue(
      (0 until A.size).forall(i => D_==(A(i), B(i), tolerance)),
      s"vectors differ beyond tolerance $tolerance",
    )
  }

  def assertMatrixEqualityDouble(
    A: Matrix[Double],
    B: Matrix[Double],
    tolerance: Double = defaultTolerance,
  ): Unit = {
    assertTrue(A.rows == B.rows && A.cols == B.cols, s"matrix dimensions differ")
    assertTrue(
      (0 until A.rows).forall(i =>
        (0 until A.cols).forall(j => D_==(A(i, j), B(i, j), tolerance))
      ),
      s"matrices differ beyond tolerance $tolerance",
    )
  }

  def isConstant(A: Vector[Int]): Boolean = {
    (0 until A.length - 1).foreach(i => if (A(i) != A(i + 1)) return false)
    true
  }

  def removeConstantCols(A: DenseMatrix[Int]): DenseMatrix[Int] = {
    val data = (0 until A.cols).flatMap { j =>
      val col = A(::, j)
      if (isConstant(col)) Array[Int]()
      else col.toArray
    }.toArray
    val newCols = data.length / A.rows
    new DenseMatrix(A.rows, newCols, data)
  }

  def unphasedDiploidGtIndicesToBoxedCall(m: DenseMatrix[Int]): DenseMatrix[BoxedCall] =
    m.map(g => if (g == -1) null: BoxedCall else Call2.fromUnphasedDiploidGtIndex(g): BoxedCall)

  def assertNDEvals(
    nd: IR,
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertNDEvals(nd, Env.empty, FastSeq(), None, expected)

  def assertNDEvals(
    nd: IR,
    expected: (Any, IndexedSeq[Long]),
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    if (expected == null)
      assertNDEvals(nd, Env.empty, FastSeq(), None, null, null)
    else
      assertNDEvals(nd, Env.empty, FastSeq(), None, expected._2, expected._1)

  def assertNDEvals(
    nd: IR,
    args: IndexedSeq[(Any, Type)],
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertNDEvals(nd, Env.empty, args, None, expected)

  def assertNDEvals(
    nd: IR,
    agg: (IndexedSeq[Row], TStruct),
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit =
    assertNDEvals(nd, Env.empty, FastSeq(), Some(agg), expected)

  def assertNDEvals(
    nd: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit = {
    var e: IndexedSeq[Any] = expected.asInstanceOf[IndexedSeq[Any]]
    val dims = ArraySeq.fill(nd.typ.asInstanceOf[TNDArray].nDims) {
      val n = e.length
      if (n != 0 && e.head.isInstanceOf[IndexedSeq[_]])
        e = e.head.asInstanceOf[IndexedSeq[Any]]
      n.toLong
    }
    assertNDEvals(nd, Env.empty, FastSeq(), agg, dims, expected)
  }

  def assertNDEvals(
    nd: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    dims: IndexedSeq[Long],
    expected: Any,
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit = {
    val arrayIR = if (expected == null) nd
    else {
      val refs = ArraySeq.fill(nd.typ.asInstanceOf[TNDArray].nDims)(Ref(freshName(), TInt32))
      bindIR(nd) { nd =>
        dims.zip(refs).foldRight[IR](NDArrayRef(nd, refs.map(Cast(_, TInt64)), -1)) {
          case ((n, ref), accum) =>
            ToArray(StreamMap(rangeIR(n.toInt), ref.name, accum))
        }
      }
    }
    assertEvalsTo(arrayIR, env, args, agg, expected)
  }

  def assertBMEvalsTo(
    bm0: BlockMatrixIR,
    expected: DenseMatrix[Double],
  )(implicit
    ctx: ExecuteContext,
    execStrats: Set[ExecStrategy],
  ): Unit = {
    val filteredExecStrats: Set[ExecStrategy] =
      if (ctx.backend.isInstanceOf[SparkBackend]) execStrats
      else {
        logger.info("skipping interpret and non-lowering compile steps on non-spark backend")
        execStrats.intersect(ExecStrategy.backendOnly)
      }

    val bm = bm0.deepCopy
    filteredExecStrats.filter(ExecStrategy.interpretOnly).foreach { strat =>
      try {
        val res = unoptimized { ctx =>
          strat match {
            case ExecStrategy.Interpret => Interpret(bm, ctx)
            case ExecStrategy.InterpretUnoptimized => Interpret(bm, ctx)
          }
        }
        assertTrue(res.toBreezeMatrix() == expected, s"block matrix result differed from expected")
      } catch {
        case e: Exception =>
          logger.error(s"error from strategy $strat", e)
          if (execStrats.contains(strat)) throw e
      }
    }
    val expectedArray = ArraySeq.tabulate(expected.rows)(i =>
      ArraySeq.tabulate(expected.cols)(j => expected(i, j))
    )
    assertNDEvals(BlockMatrixCollect(bm), expectedArray)(
      ctx,
      filteredExecStrats.filterNot(ExecStrategy.interpretOnly),
    )
  }
}
