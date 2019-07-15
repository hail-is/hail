package is.hail.expr.types.encoded

final case class EField(name: String, typ: EType, index: Int)

abstract class EBaseStruct extends EType

object EStruct {
  def apply(required: Boolean, args: (String, EType)*): EStruct =
    EStruct(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => EField(n, t, i) }
      .toArray,
      required)
}

final case class EStruct(fields: IndexedSeq[EField], override val required: Boolean = false) extends EBaseStruct {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i })
}

final case class ETuple(types: IndexedSeq[EType], override val required: Boolean = false) extends EBaseStruct
