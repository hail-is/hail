package is.hail.expr.ir

import java.security.SecureRandom

import is.hail.HailContext
import is.hail.utils._
import is.hail.annotations.Region
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.io.fs.FS

import scala.collection.mutable

object ExecuteContext {
  def scoped[T]()(f: ExecuteContext => T): T = HailContext.sparkBackend("ExecuteContext.scoped").withExecuteContext()(f)

  def scoped[T](tmpdir: String, localTmpdir: String, backend: Backend, fs: FS)(f: ExecuteContext => T): T = {
    Region.scoped { r =>
      val ctx = new ExecuteContext(tmpdir, localTmpdir, backend, fs, r, new ExecutionTimer)
      f(ctx)
    }
  }

  def scopedNewRegion[T](ctx: ExecuteContext)(f: ExecuteContext => T): T = {
    Region.scoped { r =>
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
  val timer: ExecutionTimer) {
  def fsBc: BroadcastValue[FS] = fs.broadcast

  private val tmpPaths = mutable.ArrayBuffer[String]()

  def createTmpPath(prefix: String, extension: String = null): String = {
    val path = ExecuteContext.createTmpPathNoCleanup(tmpdir, prefix, extension)
    tmpPaths += path
    path
  }

  def close(): Unit = {
    for (p <- tmpPaths)
      fs.delete(p, recursive = true)
    tmpPaths.clear()
  }
}
