package is.hail.backend

import is.hail.asm4s.HailClassLoader
import is.hail.{HailContext, HailFeatureFlags}
import is.hail.annotations.{Region, RegionPool}
import is.hail.io.fs.FS
import is.hail.utils.{ExecutionTimer, using}

import java.io._
import java.security.SecureRandom
import scala.collection.mutable

trait TempFileManager {
  def own(path: String): Unit

  def cleanup(): Unit
}

class OwningTempFileManager(fs: FS) extends TempFileManager {
  private[this] val tmpPaths = mutable.ArrayBuffer[String]()

  def own(path: String): Unit = tmpPaths += path

  override def cleanup(): Unit = {
    for (p <- tmpPaths)
      fs.delete(p, recursive = true)
    tmpPaths.clear()
  }
}

class NonOwningTempFileManager(owner: TempFileManager) extends TempFileManager {
  def own(path: String): Unit = owner.own(path)

  override def cleanup(): Unit = ()
}


object ExecuteContext {
  def scoped[T]()(f: ExecuteContext => T): T = {
    val (result, _) = ExecutionTimer.time("ExecuteContext.scoped") { timer =>
      HailContext.sparkBackend("ExecuteContext.scoped").withExecuteContext(timer, selfContainedExecution = false)(f)
    }
    result
  }

  def scoped[T](
    tmpdir: String,
    localTmpdir: String,
    backend: Backend,
    fs: FS,
    timer: ExecutionTimer,
    tempFileManager: TempFileManager,
    theHailClassLoader: HailClassLoader,
    flags: HailFeatureFlags,
  )(
    f: ExecuteContext => T
  ): T = {
    RegionPool.scoped { pool =>
      using(new ExecuteContext(
        tmpdir,
        localTmpdir,
        backend,
        fs,
        Region(pool = pool),
        timer,
        tempFileManager,
        theHailClassLoader,
        flags
      ))(f(_))
    }
  }

  def scopedNewRegion[T](ctx: ExecuteContext)(f: ExecuteContext => T): T = {
    val rp = ctx.r.pool

    rp.scopedRegion { r =>
      val oldR = ctx.r
      ctx.r = r
      val t = f(ctx)
      ctx.r = oldR
      t
    }
  }

  def createTmpPathNoCleanup(tmpdir: String, prefix: String, extension: String = null): String = {
    val random = new SecureRandom()
    val alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    val token = (0 until 22).map(_ => alphabet(random.nextInt(alphabet.length))).mkString
    if (extension != null)
      s"$tmpdir/$prefix-$token.$extension"
    else
      s"$tmpdir/$prefix-$token"
  }
}

class ExecuteContext(
  val tmpdir: String,
  val localTmpdir: String,
  val backend: Backend,
  val fs: FS,
  var r: Region,
  val timer: ExecutionTimer,
  _tempFileManager: TempFileManager,
  val theHailClassLoader: HailClassLoader,
  private[this] val flags: HailFeatureFlags
) extends Closeable {
  var backendContext: BackendContext = _

  private val tempFileManager: TempFileManager = if (_tempFileManager != null)
    _tempFileManager
  else
    new OwningTempFileManager(fs)

  def fsBc: BroadcastValue[FS] = fs.broadcast

  private val cleanupFunctions = mutable.ArrayBuffer[() => Unit]()

  private[this] val broadcasts = mutable.ArrayBuffer.empty[BroadcastValue[_]]

  val memo: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any]()

  def createTmpPath(prefix: String, extension: String = null): String = {
    val path = ExecuteContext.createTmpPathNoCleanup(tmpdir, prefix, extension)
    tempFileManager.own(path)
    path
  }

  def ownCloseable(c: Closeable): Unit = {
    cleanupFunctions += c.close
  }

  def ownCleanup(cleanupFunction: () => Unit): Unit = {
    cleanupFunctions += cleanupFunction
  }

  def getFlag(name: String): String = flags.get(name)

  def shouldWriteIRFiles(): Boolean = getFlag("write_ir_files") != null

  def shouldNotLogIR(): Boolean = flags.get("no_ir_logging") != null

  def shouldLogIR(): Boolean = !shouldNotLogIR()

  def close(): Unit = {
    tempFileManager.cleanup()

    var exception: Exception = null
    for (cleanupFunction <- cleanupFunctions) {
      try {
        cleanupFunction()
      } catch {
        case exc: Exception =>
          if (exception == null) {
            exception = new RuntimeException("ExecuteContext could not cleanup all resources")
          }
          exception.addSuppressed(exc)
      }
    }
    if (exception != null) {
      throw exception
    }
  }
}
