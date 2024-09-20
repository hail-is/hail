package is.hail

import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations._
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext}
import is.hail.backend.caching.NoCaching
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir._
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
import org.scalatestplus.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterClass, AfterMethod, BeforeClass, BeforeMethod}

object HailSuite {

  val theHailClassLoader: HailClassLoader =
    new HailClassLoader(getClass.getClassLoader)

  val flags: HailFeatureFlags =
    HailFeatureFlags.fromEnv(sys.env + ("lower" -> "1"))

  lazy val hc: HailContext = {
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
    val hc = HailContext(backend)
    hc.checkRVDKeys = true
    hc
  }
}

class HailSuite extends TestNGSuite with TestUtils {

  def hc: HailContext = HailSuite.hc

  @BeforeClass
  def initFs(): Unit = {
    val conf = new Configuration(sc.hadoopConfiguration)
    fs = new HadoopFS(new SerializableHadoopConfiguration(conf))
  }

  @AfterClass
  def closeFS(): Unit =
    hadoop.fs.FileSystem.closeAll()

  var fs: FS = _
  var pool: RegionPool = _
  private[this] var ctx_ : ExecuteContext = _

  def backend: Backend = ctx.backend
  def sc: SparkContext = backend.asSpark.sc
  def timer: ExecutionTimer = ctx.timer
  def theHailClassLoader: HailClassLoader = ctx.theHailClassLoader
  override def ctx: ExecuteContext = ctx_

  @BeforeMethod
  def setupContext(context: ITestContext): Unit = {
    pool = RegionPool()
    ctx_ = new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = hc.backend,
      fs = fs,
      r = Region(pool = pool),
      timer = new ExecutionTimer(context.getName),
      _tempFileManager = null,
      theHailClassLoader = HailSuite.theHailClassLoader,
      flags = HailSuite.flags,
      irMetadata = new IrMetadata(),
      References = ImmutableMap(ReferenceGenome.builtinReferences()),
      BlockMatrixCache = NoCaching,
      CodeCache = NoCaching,
      IrCache = NoCaching,
      CoercerCache = NoCaching,
    )
  }

  @AfterMethod
  def tearDownContext(context: ITestContext): Unit = {
    ctx_.timer.finish()
    ctx_.close()
    ctx_ = null
    pool.close()

    if (sc.isStopped)
      throw new RuntimeException(s"'${context.getName}' stopped spark context!")
  }

  def assertEvalsTo(
    x: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit = {

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

    filteredExecStrats.foreach { strat =>
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
            eval(
              x,
              env,
              args,
              agg,
              bytecodePrinter =
                Option(ctx.getFlag("jvm_bytecode_dump"))
                  .map { path =>
                    val pw = new PrintWriter(new File(path))
                    pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, x)}\n */\n\n")
                    pw
                  },
              true,
              ctx,
            )
          case ExecStrategy.JvmCompileUnoptimized =>
            assert(Forall(x, node => Compilable(node)))
            eval(
              x,
              env,
              args,
              agg,
              bytecodePrinter =
                Option(ctx.getFlag("jvm_bytecode_dump"))
                  .map { path =>
                    val pw = new PrintWriter(new File(path))
                    pw.print(s"/* JVM bytecode dump for IR:\n${Pretty(ctx, x)}\n */\n\n")
                    pw
                  },
              optimize = false,
              ctx,
            )
          case ExecStrategy.LoweredJVMCompile =>
            loweredExecute(ctx, x, env, args, agg)
        }
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
    }
  }

  def assertNDEvals(nd: IR, expected: Any)(implicit execStrats: Set[ExecStrategy]): Unit =
    assertNDEvals(nd, Env.empty, FastSeq(), None, expected)

  def assertNDEvals(
    nd: IR,
    expected: (Any, IndexedSeq[Long]),
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit =
    if (expected == null)
      assertNDEvals(nd, Env.empty, FastSeq(), None, null, null)
    else
      assertNDEvals(nd, Env.empty, FastSeq(), None, expected._2, expected._1)

  def assertNDEvals(
    nd: IR,
    args: IndexedSeq[(Any, Type)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit =
    assertNDEvals(nd, Env.empty, args, None, expected)

  def assertNDEvals(
    nd: IR,
    agg: (IndexedSeq[Row], TStruct),
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit =
    assertNDEvals(nd, Env.empty, FastSeq(), Some(agg), expected)

  def assertNDEvals(
    nd: IR,
    env: Env[(Any, Type)],
    args: IndexedSeq[(Any, Type)],
    agg: Option[(IndexedSeq[Row], TStruct)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit = {
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
  ): Unit = {
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
  ): Unit = {
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
  ): Unit =
    assertEvalsTo(MakeTuple.ordered(xs.toArray.map(_._1)), Row.fromSeq(xs.map(_._2)))

  def assertEvalsTo(
    x: IR,
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit =
    assertEvalsTo(x, Env.empty, FastSeq(), None, expected)

  def assertEvalsTo(
    x: IR,
    args: IndexedSeq[(Any, Type)],
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit =
    assertEvalsTo(x, Env.empty, args, None, expected)

  def assertEvalsTo(
    x: IR,
    agg: (IndexedSeq[Row], TStruct),
    expected: Any,
  )(implicit execStrats: Set[ExecStrategy]
  ): Unit =
    assertEvalsTo(x, Env.empty, FastSeq(), Some(agg), expected)
}
