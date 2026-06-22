package is.hail

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.spark.SparkBackend
import is.hail.collection.ImmutableMap
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.expr.ir.lowering.invariant.Flags.StrictInvariants
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.rvd.RVD
import is.hail.utils.SerializableHadoopConfiguration
import is.hail.variant.ReferenceGenome

import scala.jdk.OptionConverters.RichOptional

import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.extension.{
  AfterAllCallback, ExtensionContext, ParameterContext, ParameterResolver,
}
import org.junit.jupiter.api.extension.ExtensionContext.Namespace

/** Created once per test run, closed at the end of the test run. Holds every object with test-run
  * lifetime: the Spark backend, classloader, flags, fs, region pool, and reference genomes.
  * Per-injection state (a fresh [[ExecuteContext]] with its own Region, timer, tempFileManager,
  * irMetadata) is produced by [[newExecuteContext]]; the IR caches passed to [[newExecuteContext]]
  * are class-scoped rather than fresh per invocation, so compiled-function / persisted-IR lookups
  * are shared across tests in the same class.
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

  def newExecuteContext: ExecuteContext =
    new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = backend,
      references = references,
      fs = fs,
      r = Region(pool = pool),
      tempFileManager = new OwningTempFileManager(fs),
      theHailClassLoader = hcl,
      flags = flags,
      irMetadata = new IrMetadata(),
      // disable caching in tests
      BlockMatrixCache = ImmutableMap.empty,
      CompileCache = ImmutableMap.empty,
      PersistedIrCache = ImmutableMap.empty,
      PersistedCoercerCache = ImmutableMap.empty,
    )

  override def close(): Unit = {
    backend.spark.stop()
    backend.close()
    pool.close()
    IRFunctionRegistry.clearUserFunctions()
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

  /** Walk from `extCtx` up through its parent chain and return the first stored [[ExecuteContext]]
    * we find. Reusing an ancestor's ctx lets a parameterized-test factory share its ctx with every
    * invocation downstream, and lets `@BeforeEach` share its ctx with the following `@Test`.
    */
  private def findExistingCtx(extCtx: ExtensionContext): Option[ExecuteContext] =
    extCtx.getStore(NAMESPACE).get(CTX_KEY, classOf[ExecuteContext]) match {
      case ctx: ExecuteContext => Some(ctx)
      case _ => extCtx.getParent.toScala.flatMap(findExistingCtx)
    }

  /** Build a new [[ExecuteContext]] and register it as [[AutoCloseable]] in a scope wide enough to
    * cover every downstream use. For `@Test` / `@BeforeEach` / factory calls, that's the test
    * method's scope (so every parameterized invocation of one method reuses a single Region); for
    * class-level injection (e.g. `@BeforeAll`) it's the class scope.
    */
  private def createCtx(extCtx: ExtensionContext, s: SharedResources): ExecuteContext = {
    val ctx = s.newExecuteContext
    storageScope(extCtx).getStore(NAMESPACE).put(CTX_KEY, ctx)
    ctx
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
}

object HailExtension {
  val NAMESPACE: Namespace = Namespace.create(classOf[HailExtension])
  val SHARED_KEY = "sharedResources"
  val CTX_KEY = "executeContext"
}
