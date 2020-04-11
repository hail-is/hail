package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.backend.Backend
import is.hail.io.fs.FS
import is.hail.utils._

object ExecuteContext {
  def scoped[T]()(f: ExecuteContext => T): T = scoped(HailContext.backend, HailContext.fs)(f)

  def scoped[T](backend: Backend, fs: FS)(f: ExecuteContext => T): T = {
    Region.scoped { r =>
      val ctx = new ExecuteContext(backend, fs, r, new ExecutionTimer)
      f(ctx)
    }
  }

  def scopedNewRegion[T](ctx: ExecuteContext)(f: ExecuteContext => T): T = {
    Region.scoped { r =>
      val newCtx = new ExecuteContext(ctx.backend, ctx.fs, r, ctx.timer)
      f(newCtx)
    }
  }
}

class ExecuteContext(
  val backend: Backend,
  val fs: FS,
  val r: Region,
  val timer: ExecutionTimer)
