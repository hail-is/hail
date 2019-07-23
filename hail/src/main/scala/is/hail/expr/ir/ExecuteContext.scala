package is.hail.expr.ir

import is.hail.annotations.Region

object ExecuteContext {
  def scoped[T](f: ExecuteContext => T): T = {
    Region.scoped { r =>
      f(ExecuteContext(r))
    }
  }
}

case class ExecuteContext(r: Region)
