package is.hail

import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations._
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir._
import is.hail.expr.ir.defs.{
  BlockMatrixCollect, Cast, MakeTuple, NDArrayRef, Ref, StreamMap, ToArray,
}
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.scalatest
import org.scalatest.Inspectors.forEvery
import org.scalatestplus.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterClass, AfterSuite, BeforeClass, BeforeSuite}

object HailSuite {
  val theHailClassLoader: HailClassLoader =
    new HailClassLoader(getClass.getClassLoader)

  val flags: HailFeatureFlags =
    HailFeatureFlags.fromEnv(sys.env + ("lower" -> "1"))

  private var hc_ : HailContext = _
}

class HailSuite extends TestNGSuite with TestUtils {

  private[this] var ctx_ : ExecuteContext = _

  override def ctx: ExecuteContext = ctx_
  def backend: Backend = ctx.backend
  def fs: FS = ctx.fs
  def pool: RegionPool = ctx.r.pool
  def sc: SparkContext = ctx.backend.asSpark.sc
  def theHailClassLoader: HailClassLoader = ctx.theHailClassLoader

  def getTestResource(localPath: String): String = s"hail/test/resources/$localPath"

  @BeforeSuite
  def setupHailContext(): Unit = {
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
      skipLoggingConfiguration = true,
    )
    HailSuite.hc_ = HailContext(backend)
    HailSuite.hc_.checkRVDKeys = true
  }

  @BeforeClass
  def setupExecuteContext(): Unit = {
    val backend = HailSuite.hc_.backend.asSpark
    val conf = new Configuration(backend.sc.hadoopConfiguration)
    val fs = new HadoopFS(new SerializableHadoopConfiguration(conf))
    val pool = RegionPool()
    ctx_ = new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = backend,
      references = ReferenceGenome.builtinReferences(),
      fs = fs,
      r = Region(pool = pool),
      timer = new ExecutionTimer(getClass.getSimpleName),
      tempFileManager = new OwningTempFileManager(fs),
      theHailClassLoader = HailSuite.theHailClassLoader,
      flags = HailSuite.flags,
      irMetadata = new IrMetadata(),
      BlockMatrixCache = ImmutableMap.empty,
      CodeCache = ImmutableMap.empty,
      PersistedIrCache = ImmutableMap.empty,
      PersistedCoercerCache = ImmutableMap.empty,
    )
  }

  @AfterClass
  def tearDownExecuteContext(context: ITestContext): Unit = {
    ctx_.timer.finish()
    ctx_.close()
    ctx_.r.pool.close()
    ctx_ = null

    hadoop.fs.FileSystem.closeAll()

    if (HailSuite.hc_.backend.asSpark.sc.isStopped)
      throw new RuntimeException(s"'${context.getName}' stopped spark context!")
  }

  @AfterSuite
  def tearDownHailContext(): Unit = {
    HailSuite.hc_.stop()
    HailSuite.hc_ = null
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
