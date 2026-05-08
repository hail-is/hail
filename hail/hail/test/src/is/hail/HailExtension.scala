package is.hail

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.{BaseIR, CompileCache}
import is.hail.expr.ir.LoweredTableReader.LoweredTableReaderCoercer
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.expr.ir.lowering.invariant.Flags.StrictInvariants
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.linalg.BlockMatrix
import is.hail.rvd.RVD
import is.hail.utils.{ExecutionTimer, SerializableHadoopConfiguration}
import is.hail.variant.ReferenceGenome

import scala.collection.mutable

import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.extension.{
  AfterAllCallback, ExtensionContext, ParameterContext, ParameterResolver,
}
import org.junit.jupiter.api.extension.ExtensionContext.Namespace

/** Mutable IR caches shared across every test in a single test class. One instance is created
  * lazily at class scope (on the first [[ExecuteContext]] injection for a class) and passed into
  * every [[ExecuteContext]] built for tests in that class.
  */
final class ClassLevelIrCaches {
  val blockMatrix: mutable.Map[String, BlockMatrix] = mutable.HashMap.empty
  val compile: CompileCache = mutable.HashMap.empty
  val persistedIr: mutable.Map[Int, BaseIR] = mutable.HashMap.empty
  val persistedCoercer: mutable.Map[Any, LoweredTableReaderCoercer] = mutable.HashMap.empty
}

/** Created once per test run, closed at the end of the test run. Holds every object with test-run
  * lifetime: the Spark backend, classloader, flags, fs, region pool, and reference genomes.
  * Per-injection state (a fresh [[ExecuteContext]] with its own Region, timer, tempFileManager,
  * irMetadata) is produced by [[newExecuteContext]] and owned by [[OwnedExecuteContext]]; the IR
  * caches passed to [[newExecuteContext]] are class-scoped rather than fresh per invocation, so
  * compiled-function / persisted-IR lookups are shared across tests in the same class.
  */
final class SharedResources extends AutoCloseable {
  val hcl: HailClassLoader = new HailClassLoader(getClass.getClassLoader)

  val flags: HailFeatureFlags =
    HailFeatureFlags.fromEnv(sys.env + ("lower" -> "1") + (StrictInvariants -> "1"))

  val backend: SparkBackend = SparkBackend(
    SparkSession.builder()
      .appName("HailTest")
      .master("local[2]")
      .config("spark.unsafe.exceptionOnMemoryLeak", "true")
      .config("spark.ui.showConsoleProgress", "false")
      .config("spark.ui.enabled", "false")
      .config(SparkBackend.pySparkConf)
      .getOrCreate()
  )

  val fs: FS = new HadoopFS(
    new SerializableHadoopConfiguration(new Configuration(backend.sc.hadoopConfiguration))
  )

  val pool: RegionPool = RegionPool()

  val references: Map[String, ReferenceGenome] = ReferenceGenome.builtinReferences()

  def newExecuteContext(displayName: String, caches: ClassLevelIrCaches): ExecuteContext =
    new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = backend,
      references = references,
      fs = fs,
      r = Region(pool = pool),
      timer = new ExecutionTimer(displayName),
      tempFileManager = new OwningTempFileManager(fs),
      theHailClassLoader = hcl,
      flags = flags,
      irMetadata = new IrMetadata(),
      BlockMatrixCache = caches.blockMatrix,
      CompileCache = caches.compile,
      PersistedIrCache = caches.persistedIr,
      PersistedCoercerCache = caches.persistedCoercer,
    )

  override def close(): Unit = {
    backend.spark.stop()
    backend.close()
    pool.close()
    IRFunctionRegistry.clearUserFunctions()
  }
}

/** AutoCloseable wrapper so the JUnit store releases a scoped ExecuteContext when its owning scope
  * (test method / @BeforeEach / @MethodSource invocation) ends.
  */
final class OwnedExecuteContext(val ctx: ExecuteContext) extends AutoCloseable {
  override def close(): Unit = {
    ctx.timer.finish()
    ctx.close()
  }
}

class HailExtension extends AfterAllCallback with ParameterResolver {
  import HailExtension._

  private def shared(context: ExtensionContext): SharedResources =
    context.getRoot.getStore(Namespace.GLOBAL).getOrComputeIfAbsent(
      SHARED_KEY,
      (_: String) => {
        RVD.CheckRvdKeyOrderingForTesting = true
        new SharedResources()
      },
      classOf[SharedResources],
    )

  override def afterAll(context: ExtensionContext): Unit = {
    val s = context.getRoot.getStore(Namespace.GLOBAL).get(SHARED_KEY, classOf[SharedResources])
    if (s != null && s.backend.sc.isStopped)
      throw new RuntimeException(
        s"'${context.getDisplayName}' stopped the SparkContext!"
      )
  }

  private val supportedTypes: Set[Class[_]] =
    Set(classOf[ExecuteContext], classOf[FS], classOf[Backend], classOf[RegionPool])

  override def supportsParameter(
    paramCtx: ParameterContext,
    extCtx: ExtensionContext,
  ): Boolean =
    supportedTypes.contains(paramCtx.getParameter.getType)

  override def resolveParameter(
    paramCtx: ParameterContext,
    extCtx: ExtensionContext,
  ): AnyRef = {
    val s = shared(extCtx)
    paramCtx.getParameter.getType match {
      case c if c == classOf[FS] => s.fs
      case c if c == classOf[Backend] => s.backend
      case c if c == classOf[RegionPool] => s.pool
      case c if c == classOf[ExecuteContext] =>
        findExistingCtx(extCtx).getOrElse(createCtx(extCtx, s))
    }
  }

  /** Walk from `extCtx` up through its parent chain and return the first stored
    * [[OwnedExecuteContext]] we find. Reusing an ancestor's ctx lets a parameterized-test factory
    * share its ctx with every invocation downstream, and lets `@BeforeEach` share its ctx with the
    * following `@Test`.
    */
  private def findExistingCtx(extCtx: ExtensionContext): Option[ExecuteContext] = {
    var cur: ExtensionContext = extCtx
    while (cur != null) {
      val owned = cur.getStore(NAMESPACE).get(CTX_KEY, classOf[OwnedExecuteContext])
      if (owned != null) return Some(owned.ctx)
      cur = cur.getParent.orElse(null)
    }
    None
  }

  /** Build a new [[ExecuteContext]] and register it as [[AutoCloseable]] in a scope wide enough to
    * cover every downstream use. For `@Test` / `@BeforeEach` / factory calls, that's the test
    * method's scope (so every parameterized invocation of one method reuses a single Region); for
    * class-level injection (e.g. `@BeforeAll`) it's the class scope.
    */
  private def createCtx(extCtx: ExtensionContext, s: SharedResources): ExecuteContext = {
    val owned = new OwnedExecuteContext(
      s.newExecuteContext(extCtx.getDisplayName, classCaches(extCtx))
    )
    storageScope(extCtx).getStore(NAMESPACE).put(CTX_KEY, owned)
    owned.ctx
  }

  /** The scope at which a newly created ExecuteContext should be registered. Prefer the test
    * method's extension context (so parameterized-test invocations reuse one Region for the whole
    * method); fall back to `extCtx` when no method is in scope (e.g. `@BeforeAll`).
    */
  private def storageScope(extCtx: ExtensionContext): ExtensionContext = {
    var cur: ExtensionContext = extCtx
    while (cur != null) {
      if (cur.getTestMethod.isPresent) return cur
      cur = cur.getParent.orElse(null)
    }
    extCtx
  }

  /** Look up (or create) the [[ClassLevelIrCaches]] attached to this test class. The caches live at
    * the class-level ExtensionContext so every [[ExecuteContext]] built for a test in that class
    * shares them.
    */
  private def classCaches(extCtx: ExtensionContext): ClassLevelIrCaches =
    classScope(extCtx).getStore(NAMESPACE).getOrComputeIfAbsent(
      CACHES_KEY,
      (_: String) => new ClassLevelIrCaches,
      classOf[ClassLevelIrCaches],
    )

  /** The ExtensionContext whose element is the test class (the first ancestor with a test class but
    * no test method). Used as the anchor for class-scoped state (IR caches).
    */
  private def classScope(extCtx: ExtensionContext): ExtensionContext = {
    var cur: ExtensionContext = extCtx
    while (cur != null) {
      if (cur.getTestClass.isPresent && !cur.getTestMethod.isPresent) return cur
      cur = cur.getParent.orElse(null)
    }
    extCtx
  }
}

object HailExtension {
  val NAMESPACE: Namespace = Namespace.create(classOf[HailExtension])
  val SHARED_KEY = "sharedResources"
  val CTX_KEY = "executeContext"
  val CACHES_KEY = "irCaches"
}
