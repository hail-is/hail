package is.hail.expr.types.encoded

import is.hail.variant._

abstract class ComplexEType extends EType {
  val representation: EType
}

final case class EInterval(pointType: EType, override val required: Boolean = false) extends ComplexEType {
  val representation: EStruct = EStruct(
      required,
      "start" -> pointType,
      "end" -> pointType,
      "includesStart" -> EBooleanRequired,
      "includesEnd" -> EBooleanRequired)
}

final case class ELocus(rgBc: BroadcastRGBase, override val required: Boolean = false) extends ComplexEType {
  val representation: EStruct = EStruct(
    required,
    "contig" -> EString(required = true),
    "position" -> EInt32(required = true))
}
