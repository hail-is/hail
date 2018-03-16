package is.hail.testUtils

import is.hail.expr._
import is.hail.table.Table
import is.hail.utils._

class RichTable(ht: Table) {
  def forall(code: String): Boolean = {
    val ec = ht.rowEvalContext()
    ec.set(0, ht.globals.value)

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    ht.rdd.forall { a =>
      ec.set(1, a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def exists(code: String): Boolean = {
    val ec = ht.rowEvalContext()
    ec.set(0, ht.globals.value)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    ht.rdd.exists { a =>
      ec.set(1, a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

}
