package is.hail.expr.ir.functions

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types._

class UtilFunctions {

  val triangle: IRFunction[Int] = IRFunction[Int]("triangle", TInt32(), TInt32()) {
    case (_, Array(n: Code[Int])) => (n * (n + 1)) / 2
  }
}
