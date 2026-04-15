package is.hail

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{Backend, ExecuteContext, OwningTempFileManager}
import is.hail.backend.spark.SparkBackend
import is.hail.collection.ImmutableMap
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.rvd.RVD
import is.hail.utils.{ExecutionTimer, SerializableHadoopConfiguration}
import is.hail.variant.ReferenceGenome

import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.extension.{
  AfterAllCallback, BeforeAllCallback, ExtensionContext, ParameterContext, ParameterResolver,
}
import org.junit.jupiter.api.extension.ExtensionContext.Namespace

/** Created once per test run, closed at the end of the test run. */
final class SharedBackend extends AutoCloseable {
  val hcl: HailClassLoader = new HailClassLoader(getClass.getClassLoader)
  val flags: HailFeatureFlags = HailFeatureFlags.fromEnv(sys.env + ("lower" -> "1"))

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

  override def close(): Unit = {
    backend.spark.stop()
    backend.close()
    IRFunctionRegistry.clearUserFunctions()
  }
}

/** Created per test class, closed after the class. */
final class OwnedExecuteContext(shared: SharedBackend, displayName: String)
    extends AutoCloseable {

  val ctx: ExecuteContext = {
    val conf = new Configuration(shared.backend.sc.hadoopConfiguration)
    val fs = new HadoopFS(new SerializableHadoopConfiguration(conf))
    val pool = RegionPool()
    new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = shared.backend,
      references = ReferenceGenome.builtinReferences(),
      fs = fs,
      r = Region(pool = pool),
      timer = new ExecutionTimer(displayName),
      tempFileManager = new OwningTempFileManager(fs),
      theHailClassLoader = shared.hcl,
      flags = shared.flags,
      irMetadata = new IrMetadata(),
      BlockMatrixCache = ImmutableMap.empty,
      CompileCache = ImmutableMap.empty,
      PersistedIrCache = ImmutableMap.empty,
      PersistedCoercerCache = ImmutableMap.empty,
    )
  }

  override def close(): Unit = {
    ctx.timer.finish()
    ctx.close()
    ctx.r.pool.close()
  }
}

class HailExtension extends BeforeAllCallback with AfterAllCallback with ParameterResolver {
  import HailExtension._

  override def beforeAll(context: ExtensionContext): Unit = {
    val rootStore = context.getRoot.getStore(Namespace.GLOBAL)
    val shared = rootStore.getOrComputeIfAbsent(
      BACKEND_KEY,
      (_: String) => {
        RVD.CheckRvdKeyOrderingForTesting = true
        new SharedBackend()
      },
      classOf[SharedBackend],
    )
    context.getStore(NAMESPACE).put(
      CTX_KEY,
      new OwnedExecuteContext(shared, context.getDisplayName),
    )
  }

  override def afterAll(context: ExtensionContext): Unit = {
    val rootStore = context.getRoot.getStore(Namespace.GLOBAL)
    val shared = rootStore.get(BACKEND_KEY, classOf[SharedBackend])
    if (shared != null && shared.backend.sc.isStopped)
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
    val ctx = extCtx.getStore(NAMESPACE).get(CTX_KEY, classOf[OwnedExecuteContext]).ctx
    paramCtx.getParameter.getType match {
      case c if c == classOf[ExecuteContext] => ctx
      case c if c == classOf[FS] => ctx.fs
      case c if c == classOf[Backend] => ctx.backend
      case c if c == classOf[RegionPool] => ctx.r.pool
    }
  }
}

object HailExtension {
  val NAMESPACE: Namespace = Namespace.create(classOf[HailExtension])
  val BACKEND_KEY = "sharedBackend"
  val CTX_KEY = "executeContext"
}
