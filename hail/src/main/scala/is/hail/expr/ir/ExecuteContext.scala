package is.hail.expr.ir

import is.hail.HailContext
import is.hail.utils._
import is.hail.annotations.Region
import is.hail.backend.Backend
import is.hail.io.fs.FS

object ExecuteContext {
  def scoped[T]()(f: ExecuteContext => T): T = scoped(HailContext.backend, HailContext.fs)(f)

  def scoped[T](backend: Backend, fs: FS)(f: ExecuteContext => T): T = {
    Region.scoped { r =>
      val ctx = new ExecuteContext(backend, fs, r, new ExecutionTimer)
      f(ctx)
    }
  }
}

class ExecuteContext(
  val backend: Backend,
  val fs: FS,
  val r: Region,
  val timer: ExecutionTimer)
