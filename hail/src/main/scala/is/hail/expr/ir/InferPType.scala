package is.hail.expr.ir

import is.hail.types.TypeWithRequiredness
import is.hail.types.physical._

object InferPType {
  def getCompatiblePType(pTypes: Seq[PType]): PType = {
    val r = TypeWithRequiredness.apply(pTypes.head.virtualType)
    pTypes.foreach(r.fromPType)
    getCompatiblePType(pTypes, r)
  }

  def getCompatiblePType(pTypes: Seq[PType], result: TypeWithRequiredness): PType = {
    assert(pTypes.tail.forall(pt => pt.virtualType == pTypes.head.virtualType))
    if (pTypes.tail.forall(pt => pt == pTypes.head))
      pTypes.head
    else result.canonicalPType(pTypes.head.virtualType)
  }
}
