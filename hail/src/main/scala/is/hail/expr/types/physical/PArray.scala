package is.hail.expr.types.physical

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
import is.hail.utils._
import scala.reflect.{ClassTag, _}
object PArray {
  def apply(elementType: PType, required: Boolean = false) = new PCanonicalArray(elementType, required)
}

abstract class PArray extends PContainer with PStreamable {
  lazy val virtualType: TArray = TArray(elementType.virtualType, required)

  override lazy val fundamentalType: PArray = {
    if(this.isInstanceOf[PCanonicalArray]) {
      this
    } else {
      new PCanonicalArray(this.elementType, this.required)
    }
  }

  def copy(elementType: PType = this.elementType, required: Boolean) = new PCanonicalArray(elementType, required)
}