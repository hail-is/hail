package is.hail

import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations._
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.rvd.RVD
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import java.io.{File, PrintWriter}

import breeze.linalg.DenseMatrix
import org.apache.hadoop
import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatestplus.testng.TestNGSuite
import org.testng.ITestContext
import org.testng.annotations.{AfterClass, AfterSuite, BeforeClass, BeforeSuite}

object HailSuite {
  private val hcl: HailClassLoader =
    new HailClassLoader(getClass.getClassLoader)

  private val flags: HailFeatureFlags =
    HailFeatureFlags.fromEnv(sys.env + ("lower" -> "1"))

  private var backend_ : SparkBackend = _
}

class HailSuite extends TestNGSuite with TestUtils with Logging {

  private[this] var ctx_ : ExecuteContext = _

  override def ctx: ExecuteContext = ctx_
  def backend: Backend = ctx.backend
  def fs: FS = ctx.fs
  def pool: RegionPool = ctx.r.pool
  def sc: SparkContext = ctx.backend.asSpark.sc
  def theHailClassLoader: HailClassLoader = ctx.theHailClassLoader

  private[this] lazy val resources: String =
    sys.env.getOrElse("MILL_TEST_RESOURCE_DIR", "hail/test/resources")

  def getTestResource(localPath: String): String = s"$resources/$localPath"

  @BeforeSuite
  def setupBackend(): Unit = {
    RVD.CheckRvdKeyOrderingForTesting = true
    HailSuite.backend_ = SparkBackend(
      SparkSession
        .builder()
        .config(
          SparkBackend.createSparkConf(
            appName = "Hail.TestNG",
            master = System.getProperty("hail.master"),
            local = "local[2]",
            blockSize = 0,
          )
        )
        .config("spark.unsafe.exceptionOnMemoryLeak", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
  }

  @BeforeClass
  def setupExecuteContext(): Unit = {
    val conf = new Configuration(HailSuite.backend_.sc.hadoopConfiguration)
    val fs = new HadoopFS(new SerializableHadoopConfiguration(conf))
    val pool = RegionPool()
    ctx_ = new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = HailSuite.backend_,
      references = ReferenceGenome.builtinReferences(),
      fs = fs,
      r = Region(pool = pool),
      timer = new ExecutionTimer(getClass.getSimpleName),
      tempFileManager = new OwningTempFileManager(fs),
      theHailClassLoader = HailSuite.hcl,
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

    if (HailSuite.backend_.sc.isStopped)
      throw new RuntimeException(s"'${context.getName}' stopped spark context!")
  }

  @AfterSuite
  def tearDownBackend(): Unit = {
    HailSuite.backend_.spark.stop()
    HailSuite.backend_.close()
    HailSuite.backend_ = null
    IRFunctionRegistry.clearUserFunctions()
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
        unoptimized(ctx => Interpret[Any](ctx, ir, env, args))
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
          ctx,
        )
      case ExecStrategy.JvmCompileUnoptimized =>
        assert(Forall(ir, node => Compilable(node)))
        unoptimized { ctx =>
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
            ctx,
          )
        }
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
  ): Unit = {

    TypeCheck(ctx, x, BindingEnv(env.mapValues(_._2), agg = agg.map(_._2.toEnv)))

    val t = x.typ
    assert(t == TVoid || t.typeCheck(expected), s"$t, $expected")

    val filteredExecStrats: Set[ExecStrategy] =
      if (backend.isInstanceOf[SparkBackend])
        execStrats
      else {
        logger.info("skipping interpret and non-lowering compile steps on non-spark backend")
        execStrats.intersect(ExecStrategy.backendOnly)
      }

    filteredExecStrats.foreach { implicit strat =>
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
          logger.error(s"error from strategy $strat", e)
          if (execStrats.contains(strat)) throw e
      }

      succeed
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
      if (backend.isInstanceOf[SparkBackend]) execStrats
      else {
        logger.info("skipping interpret and non-lowering compile steps on non-spark backend")
        execStrats.intersect(ExecStrategy.backendOnly)
      }
    filteredExecStrats.filter(ExecStrategy.interpretOnly).foreach { strat =>
      try {
        val res =
          unoptimized { ctx =>
            strat match {
              case ExecStrategy.Interpret =>
                Interpret(bm, ctx)
              case ExecStrategy.InterpretUnoptimized =>
                Interpret(bm, ctx)
            }
          }
        assert(res.toBreezeMatrix() == expected)
      } catch {
        case e: Exception =>
          logger.error(s"error from strategy $strat")
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
