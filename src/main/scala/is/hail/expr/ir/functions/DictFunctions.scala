package is.hail.expr.ir.functions

import is.hail.expr.ir._
import is.hail.expr.types._

object DictFunctions extends RegistryFunctions {

  def registerAll() {
    registerIR("toDict", TArray(tv("T")))(ToDict)

    registerIR("size", TDict(tv("T"), tv("U"))) { d =>
      ArrayLen(ToArray(d))
    }

    registerIR("isEmpty", TDict(tv("T"), tv("U"))) { d =>
      ArrayFunctions.isEmpty(ToArray(d))
    }
  }
}
