package is.hail

import breeze.linalg.DenseMatrix
import cats.implicits.catsSyntaxApply
import cats.mtl.Ask
import cats.{Applicative, MonadThrow}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations._
import is.hail.backend.spark.SparkBackend
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.utils._
import is.hail.expr.ir.lowering.{Lower, LoweringState}
import is.hail.io.fs.FS
import is.hail.types.virtual._
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterMethod, BeforeClass, BeforeMethod}

import java.io.{File, PrintWriter}
import scala.annotation.tailrec
import scala.language.{higherKinds, implicitConversions}
import scala.util.control.NonFatal

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
          blockSize = 0)
          .set("spark.unsafe.exceptionOnMemoryLeak", "true")),
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      skipLoggingConfiguration = true)
    HailContext(backend)
  }

  lazy val hc: HailContext = {
    val hc = withSparkBackend()
    hc.sparkBackend("HailSuite.hc").setFlag("lower", "1")
    hc.checkRVDKeys = true
    hc
  }
}

class HailSuite extends TestNGSuite {
  val theHailClassLoader = HailSuite.theHailClassLoader

  def hc: HailContext = HailSuite.hc

  @BeforeClass def ensureHailContextInitialized() { hc }

  def backend: SparkBackend = hc.sparkBackend("HailSuite.backend")

  def sc: SparkContext = backend.sc

  def fs: FS = backend.fs

  def fsBc: BroadcastValue[FS] = fs.broadcast

  var timer: ExecutionTimer = _

  var ctx: ExecuteContext = _

  var pool: RegionPool = _

  @BeforeMethod
  def setupContext(context: ITestContext): Unit = {
    assert(timer == null)
    timer = new ExecutionTimer("HailSuite")
    assert(ctx == null)
    pool = RegionPool()
    ctx = backend.createExecuteContextForTests(timer, Region(pool=pool))
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

  def withExecuteContext[T]()(f: ExecuteContext => T): T = {
    ExecutionTimer.logTime("HailSuite.withExecuteContext") { timer =>
      hc.sparkBackend("HailSuite.withExecuteContext").withExecuteContext(timer)(f)
    }
  }

  def assertEvalsTo(
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any
  )(
    implicit execStrats: Set[ExecStrategy]
  ) {

    TypeCheck(ctx, x, BindingEnv(env.mapValues(_._2), agg = agg.map(_._2.toEnv)))

    val t = x.typ
    assert(t == TVoid || t.typeCheck(expected), s"$t, $expected")

    ExecuteContext.scoped() { ctx =>
      val filteredExecStrats: Set[ExecStrategy] =
        if (HailContext.backend.isInstanceOf[SparkBackend])
          execStrats
        else {
          info("skipping interpret and non-lowering compile steps on non-spark backend")
          execStrats.intersect(ExecStrategy.backendOnly)
        }

      import Lower.monadLowerInstanceForLower
      filteredExecStrats.foreach { strat =>
        val res = (strat match {
          case ExecStrategy.Interpret =>
            assertA(agg.isEmpty) *> Interpret(x, env, args)
          case ExecStrategy.InterpretUnoptimized =>
            assertA(agg.isEmpty) *> Interpret(x, env, args, optimize = false)
          case ExecStrategy.JvmCompile =>
            assertA(Forall(x, node => Compilable(node))) *> eval(x, env, args, agg,
              bytecodePrinter = Option(ctx.getFlag("jvm_bytecode_dump"))
                .map { path =>
                  val pw = new PrintWriter(new File(path))
                  pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, x)}\n */\n\n")
                  pw
                }
            )
          case ExecStrategy.JvmCompileUnoptimized =>
            assertA(Forall(x, node => Compilable(node))) *> eval(x, env, args, agg,
              bytecodePrinter = Option(ctx.getFlag("jvm_bytecode_dump"))
                .map { path =>
                  val pw = new PrintWriter(new File(path))
                  pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, x)}\n */\n\n")
                  pw
                },
              optimize = false
            )
          case ExecStrategy.LoweredJVMCompile =>
            Lower.Pure(loweredExecute(ctx, x, env, args, agg))
        }).runA(ctx, LoweringState())
        if (t != TVoid) {
          assert(t.typeCheck(res), s"\n  t=$t\n  result=$res\n  strategy=$strat")
          assert(t.valuesSimilar(res, expected), s"\n  result=$res\n  expect=$expected\n  strategy=$strat)")
        }
      }
    }
  }

  def assertNDEvals(nd: IR, expected: Any)(implicit execStrats: Set[ExecStrategy]): Unit =
    assertNDEvals(nd, Env.empty, FastIndexedSeq(), None, expected)

  def assertNDEvals(nd: IR, expected: (Any, IndexedSeq[Long]))
                   (implicit execStrats: Set[ExecStrategy]): Unit =
    if (expected == null)
      assertNDEvals(nd, Env.empty, FastIndexedSeq(), None, null, null)
    else
      assertNDEvals(nd, Env.empty, FastIndexedSeq(), None, expected._2, expected._1)

  def assertNDEvals(nd: IR, args: IndexedSeq[(Any, Type)], expected: Any)
                   (implicit execStrats: Set[ExecStrategy]): Unit =
    assertNDEvals(nd, Env.empty, args, None, expected)

  def assertNDEvals(nd: IR, agg: (IndexedSeq[Row], TStruct), expected: Any)
                   (implicit execStrats: Set[ExecStrategy]): Unit =
    assertNDEvals(nd, Env.empty, FastIndexedSeq(), Some(agg), expected)


  def assertNDEvals(
    nd: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any
  )(
    implicit execStrats: Set[ExecStrategy]
  ): Unit = {
    var e: IndexedSeq[Any] = expected.asInstanceOf[IndexedSeq[Any]]
    val dims = Array.fill(nd.typ.asInstanceOf[TNDArray].nDims) {
      val n = e.length
      if (n != 0 && e.head.isInstanceOf[IndexedSeq[_]])
        e = e.head.asInstanceOf[IndexedSeq[Any]]
      n.toLong
    }
    assertNDEvals(nd, Env.empty, FastIndexedSeq(), agg, dims, expected)
  }

  def assertNDEvals(
    nd: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    dims: IndexedSeq[Long],
    expected: Any
  )(
    implicit execStrats: Set[ExecStrategy]
  ): Unit = {
    val arrayIR = if (expected == null) nd else {
      val refs = Array.fill(nd.typ.asInstanceOf[TNDArray].nDims) { Ref(genUID(), TInt32) }
      Let("nd", nd,
        dims.zip(refs).foldRight[IR](NDArrayRef(Ref("nd", nd.typ), refs.map(Cast(_, TInt64)), -1)) {
          case ((n, ref), accum) =>
            ToArray(StreamMap(rangeIR(n.toInt), ref.name, accum))
        })
    }
    assertEvalsTo(arrayIR, env, args, agg, expected)
  }

  def assertBMEvalsTo(bm: BlockMatrixIR, expected: DenseMatrix[Double])
                     (implicit execStrats: Set[ExecStrategy]): Unit = {
    ExecuteContext.scoped() { ctx =>
      val filteredExecStrats: Set[ExecStrategy] =
        if (HailContext.backend.isInstanceOf[SparkBackend]) execStrats
        else {
          info("skipping interpret and non-lowering compile steps on non-spark backend")
          execStrats.intersect(ExecStrategy.backendOnly)
        }
      filteredExecStrats.filter(ExecStrategy.interpretOnly).foreach { strat =>
        import Lower.monadLowerInstanceForLower
        assert(
          Interpret(bm, optimize = strat == ExecStrategy.Interpret)
            .runA(ctx, LoweringState())
            .toBreezeMatrix() == expected
        )
      }
      val expectedArray = Array.tabulate(expected.rows)(i => Array.tabulate(expected.cols)(j => expected(i, j)).toFastIndexedSeq).toFastIndexedSeq
      assertNDEvals(BlockMatrixCollect(bm), expectedArray)(filteredExecStrats.filterNot(ExecStrategy.interpretOnly))
    }
  }

  def assertAllEvalTo(xs: (IR, Any)*)(implicit execStrats: Set[ExecStrategy]): Unit =
    assertEvalsTo(MakeTuple.ordered(xs.toArray.map(_._1)), Row.fromSeq(xs.map(_._2)))

  def assertEvalsTo(x: IR, expected: Any)(implicit execStrats: Set[ExecStrategy]): Unit =
    assertEvalsTo(x, Env.empty, FastIndexedSeq(), None, expected)

  def assertEvalsTo(x: IR, args: IndexedSeq[(Any, Type)], expected: Any)
                   (implicit execStrats: Set[ExecStrategy]): Unit =
    assertEvalsTo(x, Env.empty, args, None, expected)

  def assertEvalsTo(x: IR, agg: (IndexedSeq[Row], TStruct), expected: Any)
                   (implicit execStrats: Set[ExecStrategy]): Unit =
    assertEvalsTo(x, Env.empty, FastIndexedSeq(), Some(agg), expected)

}

trait MonadRunSupport {
  type Run[A] = ExecuteContext => A

  implicit object monadThrowInstanceForRun extends MonadThrow[Run] {
    override def raiseError[A](e: Throwable): Run[A] = {
      val t = e.fillInStackTrace()
      _ => throw t
    }

    override def handleErrorWith[A](fa: Run[A])(f: Throwable => Run[A]): Run[A] =
      ctx => try {
        fa(ctx)
      } catch {
        case NonFatal(t) => f(t)(ctx)
      }

    override def flatMap[A, B](fa: Run[A])(f: A => Run[B]): Run[B] =
      ctx => f(fa(ctx))(ctx)

    override def tailRecM[A, B](a0: A)(f: A => Run[Either[A, B]]): Run[B] = {
      ctx =>
        @tailrec def go(a: A): B = f(a)(ctx) match {
          case Left(a) => go(a)
          case Right(b) => b
        }

        go(a0)
    }

    override def pure[A](x: A): Run[A] =
      _ => x
  }

  implicit object askInstanceForRun extends Ask[Run, ExecuteContext] {
    override def applicative: Applicative[Run] = monadThrowInstanceForRun
    override def ask[E2 >: ExecuteContext]: Run[E2] = identity
  }
}
