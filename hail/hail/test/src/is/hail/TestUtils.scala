package is.hail

import is.hail.annotations.{Region, RegionPool, RegionValueBuilder, SafeRow}
import is.hail.asm4s.{implicits => _, _}
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.{
  freshName, streamAggIR, BindingEnv, Env, IR, Interpret, MapIR, MatrixIR, MatrixRead, Name,
  SingleCodeEmitParamType, Subst,
}
import is.hail.expr.ir.Optimize.Flags.Optimize
import is.hail.expr.ir.compile.Compile
import is.hail.expr.ir.defs.{GetField, GetTupleElement, In, MakeTuple, Ref, ToStream}
import is.hail.expr.ir.lowering.LowererUnsupportedOperation
import is.hail.io.vcf.MatrixVCFReader
import is.hail.types.physical.{PBaseStruct, PCanonicalArray, PType}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._

import java.io.PrintWriter

import breeze.linalg.{DenseMatrix, Matrix, Vector}
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.scalatest.{Assertion, Assertions}

object ExecStrategy extends Enumeration {
  type ExecStrategy = Value
  val Interpret, InterpretUnoptimized, JvmCompile, LoweredJVMCompile, JvmCompileUnoptimized = Value

  val unoptimizedCompileOnly: Set[ExecStrategy] = Set(JvmCompileUnoptimized)
  val compileOnly: Set[ExecStrategy] = Set(JvmCompile, JvmCompileUnoptimized)

  val javaOnly: Set[ExecStrategy] =
    Set(Interpret, InterpretUnoptimized, JvmCompile, JvmCompileUnoptimized)

  val interpretOnly: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized)

  val nonLowering: Set[ExecStrategy] =
    Set(Interpret, InterpretUnoptimized, JvmCompile, JvmCompileUnoptimized)

  val lowering: Set[ExecStrategy] = Set(LoweredJVMCompile)
  val backendOnly: Set[ExecStrategy] = Set(LoweredJVMCompile)
  val allRelational: Set[ExecStrategy] = interpretOnly.union(lowering)
}

trait TestUtils extends Assertions {

  def ctx: ExecuteContext = ???

  def interceptException[E <: Throwable: Manifest](regex: String)(f: => Any): Assertion = {
    val thrown = intercept[E](f)
    val p = regex.r.findFirstIn(thrown.getMessage).isDefined
    val msg =
      s"""expected fatal exception with pattern '$regex'
         |  Found: ${thrown.getMessage} """
    if (!p)
      println(msg)
    assert(p, msg)
  }

  def interceptFatal(regex: String)(f: => Any): Assertion =
    interceptException[HailException](regex)(f)

  def interceptSpark(regex: String)(f: => Any): Assertion =
    interceptException[SparkException](regex)(f)

  def interceptAssertion(regex: String)(f: => Any): Assertion =
    interceptException[AssertionError](regex)(f)

  def assertVectorEqualityDouble(
    A: Vector[Double],
    B: Vector[Double],
    tolerance: Double = utils.defaultTolerance,
  ): Assertion = {
    assert(A.size == B.size)
    assert((0 until A.size).forall(i => D_==(A(i), B(i), tolerance)))
  }

  def assertMatrixEqualityDouble(
    A: Matrix[Double],
    B: Matrix[Double],
    tolerance: Double = utils.defaultTolerance,
  ): Assertion = {
    assert(A.rows == B.rows)
    assert(A.cols == B.cols)
    assert((0 until A.rows).forall(i =>
      (0 until A.cols).forall(j => D_==(A(i, j), B(i, j), tolerance))
    ))
  }

  def isConstant(A: Vector[Int]): Boolean = {
    (0 until A.length - 1).foreach(i => if (A(i) != A(i + 1)) return false)
    true
  }

  def removeConstantCols(A: DenseMatrix[Int]): DenseMatrix[Int] = {
    val data = (0 until A.cols).flatMap { j =>
      val col = A(::, j)
      if (isConstant(col))
        Array[Int]()
      else
        col.toArray
    }.toArray

    val newCols = data.length / A.rows
    new DenseMatrix(A.rows, newCols, data)
  }

  def unphasedDiploidGtIndicesToBoxedCall(m: DenseMatrix[Int]): DenseMatrix[BoxedCall] =
    m.map(g => if (g == -1) null: BoxedCall else Call2.fromUnphasedDiploidGtIndex(g): BoxedCall)

  def loweredExecute(
    ctx: ExecuteContext,
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    bytecodePrinter: Option[PrintWriter] = None,
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
    ctx: ExecuteContext = ctx,
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

    def rewrite(x: IR): IR = {
      x match {
        case In(i, _) =>
          GetTupleElement(argsVar, i)
        case _ =>
          MapIR(rewrite)(x)
      }
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

  def assertEvalSame(x: IR): Assertion =
    assertEvalSame(x, Env.empty, FastSeq())

  def assertEvalSame(x: IR, args: IndexedSeq[(Any, Type)]): Assertion =
    assertEvalSame(x, Env.empty, args)

  def assertEvalSame(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)]): Assertion = {
    val t = x.typ

    val (i, i2, c) =
      ctx.local() { ctx =>
        ctx.flags.set(Optimize, "1")
        val i = Interpret[Any](ctx, x, env, args)
        ctx.flags.set(Optimize, null)
        val i2 = Interpret[Any](ctx, x, env, args)
        ctx.flags.set(Optimize, "1")
        val c = eval(x, env, args, None, None, ctx)
        (i, i2, c)
      }

    assert(t.typeCheck(i))
    assert(t.typeCheck(i2))
    assert(t.typeCheck(c))

    assert(t.valuesSimilar(i, c), s"interpret $i vs compile $c")
    assert(t.valuesSimilar(i2, c), s"interpret (optimize = false) $i vs compile $c")
  }

  def assertThrows[E <: Throwable: Manifest](x: IR, regex: String): Assertion =
    assertThrows[E](x, Env.empty[(Any, Type)], FastSeq.empty[(Any, Type)], regex)

  def assertThrows[E <: Throwable: Manifest](
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    regex: String,
  ): Assertion =
    ctx.local() { ctx =>
      ctx.flags.set(Optimize, "1")
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args))
      ctx.flags.set(Optimize, null)
      interceptException[E](regex)(Interpret[Any](ctx, x, env, args))
      ctx.flags.set(Optimize, "1")
      interceptException[E](regex)(eval(x, env, args, None, None, ctx))
    }

  def assertFatal(x: IR, regex: String): Assertion =
    assertThrows[HailException](x, regex)

  def assertFatal(x: IR, args: IndexedSeq[(Any, Type)], regex: String): Assertion =
    assertThrows[HailException](x, Env.empty[(Any, Type)], args, regex)

  def assertFatal(x: IR, env: Env[(Any, Type)], args: IndexedSeq[(Any, Type)], regex: String)
    : Assertion =
    assertThrows[HailException](x, env, args, regex)

  def assertCompiledThrows[E <: Throwable: Manifest](
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    regex: String,
  ): Assertion =
    interceptException[E](regex)(eval(x, env, args, None, None, ctx))

  def assertCompiledThrows[E <: Throwable: Manifest](x: IR, regex: String): Assertion =
    assertCompiledThrows[E](x, Env.empty[(Any, Type)], FastSeq.empty[(Any, Type)], regex)

  def assertCompiledFatal(x: IR, regex: String): Assertion =
    assertCompiledThrows[HailException](x, regex)

  def importVCF(
    ctx: ExecuteContext,
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
  ): MatrixIR = {
    val entryFloatType = TFloat64._toPretty

    val reader = MatrixVCFReader(
      ctx,
      Array(file),
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

  def cartesian[A, B](as: Iterable[A], bs: Iterable[B]): Iterable[(A, B)] =
    for {
      a <- as
      b <- bs
    } yield (a, b)

  def measuringHighestTotalMemoryUsage[A](f: => ExecuteContext => A): (A, Long) =
    RegionPool.scoped { p =>
      val a = ctx.local(r = Region(pool = p))(f)
      (a, p.getHighestTotalUsage)
    }

  def unoptimized[A](f: ExecuteContext => A): A =
    unoptimized(ctx)(f)

  def unoptimized[A](ctx: ExecuteContext)(f: ExecuteContext => A): A =
    ctx.local(flags = ctx.flags - Optimize)(f)
}
