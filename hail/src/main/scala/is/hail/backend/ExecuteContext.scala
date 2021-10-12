package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.{Region, RegionPool}
import is.hail.io.fs.FS
import is.hail.utils.{ExecutionTimer, using, log}

import java.io._
import java.security.SecureRandom
import scala.collection.mutable
import scala.reflect.ClassTag

trait TempFileManager {
  def own(path: String): Unit

  def cleanup(): Unit

  def ownBroadcast(bc: BroadcastValue[_]): Unit
}

class OwningTempFileManager(fs: FS) extends TempFileManager {
  private[this] val tmpPaths = mutable.ArrayBuffer[String]()
  private[this] val broadcasts = mutable.ArrayBuffer[BroadcastValue[_]]()


  def own(path: String): Unit = tmpPaths += path
  def ownBroadcast(bc: BroadcastValue[_]): Unit = broadcasts += bc

  override def cleanup(): Unit = {
    for (p <- tmpPaths)
      fs.delete(p, recursive = true)
    tmpPaths.clear()

    log.info(s"cleaning up ${ broadcasts.length } broadcasts")
    broadcasts.foreach { bc => bc.cleanup() }
    broadcasts.clear()
    log.info(s"done cleaning up broadcasts")
  }
}

class NonOwningTempFileManager(owner: TempFileManager) extends TempFileManager {
  private[this] val broadcasts = mutable.ArrayBuffer[BroadcastValue[_]]()

  def own(path: String): Unit = owner.own(path)

  def ownBroadcast(bc: BroadcastValue[_]): Unit = broadcasts += bc

  override def cleanup(): Unit = ()
}


object ExecuteContext {
  def scoped[T]()(f: ExecuteContext => T): T = {
    val (result, _) = ExecutionTimer.time("ExecuteContext.scoped") { timer =>
      HailContext.sparkBackend("ExecuteContext.scoped").withExecuteContext(timer, selfContainedExecution = false)(f)
    }
    result
  }

  def scoped[T](tmpdir: String, localTmpdir: String, backend: Backend, fs: FS, timer: ExecutionTimer, tempFileManager: TempFileManager)(f: ExecuteContext => T): T = {
    RegionPool.scoped { pool =>
      using(new ExecuteContext(tmpdir, localTmpdir, backend, fs, Region(pool = pool), timer, tempFileManager))(f(_))
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
  _tempFileManager: TempFileManager
) extends Closeable {
  var backendContext: BackendContext = _

  val printIRs: Boolean = HailContext.getFlag("no_ir_logging") == null

  private val tempFileManager: TempFileManager = if (_tempFileManager != null)
    _tempFileManager
  else
    new OwningTempFileManager(fs)

  def fsBc: BroadcastValue[FS] = fs.broadcast

  private val cleanupFunctions = mutable.ArrayBuffer[() => Unit]()

  val memo: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any]()

  def createTmpPath(prefix: String, extension: String = null): String = {
    val path = ExecuteContext.createTmpPathNoCleanup(tmpdir, prefix, extension)
    tempFileManager.own(path)
    path
  }

  def broadcast[T: ClassTag](value: T): BroadcastValue[T] = {
    val bc = backend.broadcast(value)
    tempFileManager.ownBroadcast(bc)
    bc
  }

  def ownCloseable(c: Closeable): Unit = {
    cleanupFunctions += c.close
  }

  def ownCleanup(cleanupFunction: () => Unit): Unit = {
    cleanupFunctions += cleanupFunction
  }

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