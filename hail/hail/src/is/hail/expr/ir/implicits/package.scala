package is.hail.expr.ir

package object implicits {
  @inline implicit def toCompiledOps[A](fa: Compiled[A]): CompiledOps[A] =
    new CompiledOps[A](fa)
}
