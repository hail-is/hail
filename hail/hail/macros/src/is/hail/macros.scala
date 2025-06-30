package is.hail

import scala.language.experimental.{macros => scalamacros}
import scala.reflect.macros.blackbox

package object macros extends VoidImpl {
  def void[A](a: A): Unit = macro implVoid[A]
}

sealed trait VoidImpl {
  def implVoid[A](c: blackbox.Context)(a: c.Expr[A]): c.Expr[Unit] = {
    import c.universe._
    c.Expr[Unit](q"""{ val _ = $a }""")
  }
}
