package is.hail

import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations._
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{
  bindIR, freshName, rangeIR, BindingEnv, BlockMatrixIR, Compilable, Env, Forall, IR, Interpret,
  Pretty, TypeCheck,
}
import is.hail.expr.ir.defs.{
  BlockMatrixCollect, Cast, MakeTuple, NDArrayRef, Ref, StreamMap, ToArray,
}
import is.hail.io.fs.FS
import is.hail.macros.void
import is.hail.types.virtual._
import is.hail.utils._

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.scalatest
import org.scalatest.Inspectors.forEvery
import org.scalatestplus.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterMethod, BeforeClass, BeforeMethod}

object HailSuite {
  val theHailClassLoader = TestUtils.theHailClassLoader

  def withSparkBackend(): HailContext = {
    HailContext.configureLogging("/tmp/hail.log", quiet = false, append = false)
    val backend = SparkBackend(
      sc = new SparkContext(
        SparkBackend.createSparkConf(
          appName = "Hail.TestNG",
          master = System.getProperty("hail.master"),
          local = "local[2]",
          blockSize = 0,
        )
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")
      ),
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      skipLoggingConfiguration = true,
    )
    HailContext(backend)
  }

  lazy val hc: HailContext = {
    val hc = withSparkBackend()
    hc.sparkBackend("HailSuite.hc").flags.set("lower", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class HailSuite extends TestNGSuite {
  val theHailClassLoader = HailSuite.theHailClassLoader

  def hc: HailContext = HailSuite.hc

  @BeforeClass def ensureHailContextInitialized(): Unit =
    void(hc)

  def backend: SparkBackend = hc.sparkBackend("HailSuite.backend")

  def sc: SparkContext = backend.sc

  def fs: FS = backend.fs

  def fsBc: BroadcastValue[FS] = fs.broadcast

  var timer: ExecutionTimer = _

  var ctx: ExecuteContext = _

  var pool: RegionPool = _

  def getTestResource(localPath: String): String = s"hail/test/resources/$localPath"

  @BeforeMethod
  def setupContext(context: ITestContext): Unit = {
    assert(timer == null)
    timer = new ExecutionTimer("HailSuite")
    assert(ctx == null)
    pool = RegionPool()
    ctx = backend.createExecuteContextForTests(timer, Region(pool = pool))
  }

  @AfterMethod
  def tearDownContext(context: ITestContext): Unit = {
    ctx.close()
    ctx = null
    timer.finish()
    timer = null
    pool.close()

    if (backend.sc.isStopped)
      throw new RuntimeException(s"method stopped spark context!")
  }

  def evaluate(
    ctx: ExecuteContext,
    ir: IR,
    args: IndexedSeq[(Any, Type)],
    env: Env[(Any, Type)] = Env.empty,
    agg: Option[(IndexedSeq[Row], TStruct)] = None,
  )(implicit strat: ExecStrategy
  ): Any =
    strat match {
      case ExecStrategy.Interpret =>
        assert(agg.isEmpty)
        Interpret[Any](ctx, ir, env, args)
      case ExecStrategy.InterpretUnoptimized =>
        assert(agg.isEmpty)
        Interpret[Any](ctx, ir, env, args, optimize = false)
      case ExecStrategy.JvmCompile =>
        assert(Forall(ir, node => Compilable(node)))
        eval(
          ir,
          env,
          args,
          agg,
          bytecodePrinter =
            Option(ctx.getFlag("jvm_bytecode_dump"))
              .map { path =>
                val pw = new PrintWriter(new File(path))
                pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, ir)}\n */\n\n")
                pw
              },
          true,
          ctx,
        )
      case ExecStrategy.JvmCompileUnoptimized =>
        assert(Forall(ir, node => Compilable(node)))
        eval(
          ir,
          env,
          args,
          agg,
          bytecodePrinter =
            Option(ctx.getFlag("jvm_bytecode_dump"))
              .map { path =>
                val pw = new PrintWriter(new File(path))
                pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, ir)}\n */\n\n")
                pw
              },
          optimize = false,
          ctx,
        )
      case ExecStrategy.LoweredJVMCompile =>
        loweredExecute(ctx, ir, env, args, agg)
    }

  def assertEvalsTo(
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion = {

    TypeCheck(ctx, x, BindingEnv(env.mapValues(_._2), agg = agg.map(_._2.toEnv)))

    val t = x.typ
    assert(t == TVoid || t.typeCheck(expected), s"$t, $expected")

    ExecuteContext.scoped { ctx =>
      val filteredExecStrats: Set[ExecStrategy] =
        if (HailContext.backend.isInstanceOf[SparkBackend])
          execStrats
        else {
          info("skipping interpret and non-lowering compile steps on non-spark backend")
          execStrats.intersect(ExecStrategy.backendOnly)
        }

      forEvery(filteredExecStrats) { implicit strat =>
        try {
          val res =
            evaluate(ctx, x, args, env, agg)

          if (t != TVoid) {
            assert(t.typeCheck(res), s"\n  t=$t\n  result=$res\n  strategy=$strat")
            assert(
              t.valuesSimilar(res, expected),
              s"\n  result=$res\n  expect=$expected\n  strategy=$strat)",
            )
          }

        } catch {
          case e: Exception =>
            error(s"error from strategy $strat")
            if (execStrats.contains(strat)) throw e
        }

        succeed
      }
    }
  }

  def assertNDEvals(nd: IR, expected: Any)(implicit execStrats: Set[ExecStrategy])
    : scalatest.Assertion =
    assertNDEvals(nd, Env.empty, FastSeq(), None, expected)

  def assertNDEvals(
    nd: IR,
    expected: (Any, IndexedSeq[Long]),
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    if (expected == null)
      assertNDEvals(nd, Env.empty, FastSeq(), None, null, null)
    else
      assertNDEvals(nd, Env.empty, FastSeq(), None, expected._2, expected._1)

  def assertNDEvals(
    nd: IR,
    args: IndexedSeq[(Any, Type)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    assertNDEvals(nd, Env.empty, args, None, expected)

  def assertNDEvals(
    nd: IR,
    agg: (IndexedSeq[Row], TStruct),
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    assertNDEvals(nd, Env.empty, FastSeq(), Some(agg), expected)

  def assertNDEvals(
    nd: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion = {
    var e: IndexedSeq[Any] = expected.asInstanceOf[IndexedSeq[Any]]
    val dims = Array.fill(nd.typ.asInstanceOf[TNDArray].nDims) {
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
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion = {
    val arrayIR = if (expected == null) nd
    else {
      val refs = Array.fill(nd.typ.asInstanceOf[TNDArray].nDims)(Ref(freshName(), TInt32))
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
    bm: BlockMatrixIR,
    expected: DenseMatrix[Double],
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion = {
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
      val expectedArray = Array.tabulate(expected.rows)(i =>
        Array.tabulate(expected.cols)(j => expected(i, j)).toFastSeq
      ).toFastSeq
      assertNDEvals(BlockMatrixCollect(bm), expectedArray)(
        filteredExecStrats.filterNot(ExecStrategy.interpretOnly)
      )
    }
  }

  def assertAllEvalTo(
    xs: (IR, Any)*
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    assertEvalsTo(MakeTuple.ordered(xs.toArray.map(_._1)), Row.fromSeq(xs.map(_._2)))

  def assertEvalsTo(
    x: IR,
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    assertEvalsTo(x, Env.empty, FastSeq(), None, expected)

  def assertEvalsTo(
    x: IR,
    args: IndexedSeq[(Any, Type)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    assertEvalsTo(x, Env.empty, args, None, expected)

  def assertEvalsTo(
    x: IR,
    agg: (IndexedSeq[Row], TStruct),
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): scalatest.Assertion =
    assertEvalsTo(x, Env.empty, FastSeq(), Some(agg), expected)
}
