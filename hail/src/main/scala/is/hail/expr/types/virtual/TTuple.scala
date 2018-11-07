package is.hail.expr.types.virtual

import is.hail.annotations.ExtendedOrdering
import is.hail.expr.types.physical.PTuple
import is.hail.utils._

import scala.collection.JavaConverters._

object TTuple {
  private val requiredEmpty = TTuple(Array.empty[Type], true)
  private val optionalEmpty = TTuple(Array.empty[Type], false)

  def empty(required: Boolean = false): TTuple = if (required) requiredEmpty else optionalEmpty

  def apply(required: Boolean, args: Type*): TTuple = TTuple(args.toArray, required)

  def apply(args: Type*): TTuple = apply(false, args: _*)

  def apply(types: java.util.ArrayList[Type], required: Boolean): TTuple = {
    val t = TTuple(types.asScala.toArray)
    t.setRequired(required).asInstanceOf[TTuple]
  }
}

final case class TTuple(_types: IndexedSeq[Type], override val required: Boolean = false) extends TBaseStruct {
  lazy val physicalType: PTuple = PTuple(types.map(_.physicalType), required)

  val types = _types.toArray
  val fieldRequired: Array[Boolean] = types.map(_.required)

  val fields: IndexedSeq[Field] = types.zipWithIndex.map { case (t, i) => Field(s"$i", t, i) }

  val ordering: ExtendedOrdering = TBaseStruct.getOrdering(types)

  val size: Int = types.length

  override def truncate(newSize: Int): TTuple =
    TTuple(types.take(newSize), required)

  val missingIdx = new Array[Int](size)
  val nMissing: Int = TBaseStruct.getMissingness(types, missingIdx)

  def ++(that: TTuple): TTuple = TTuple(types ++ that.types, required = false)

  override def canCompare(other: Type): Boolean = other match {
    case t: TTuple => size == t.size && types.zip(t.types).forall { case (t1, t2) => t1.canCompare(t2) }
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TTuple(ctypes, _) =>
      size == ctypes.length &&
        (types, ctypes).zipped.forall { case (t, ct) =>
          t.unify(ct)
        }
    case _ => false
  }

  override def subst() = TTuple(types.map(t => t.subst()))

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append("Tuple[")
    types.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
    sb += ']'
  }

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("tuple(")
    fields.foreachBetween({ field =>
      field.typ.pyString(sb)
    }) { sb.append(", ")}
    sb.append(')')
  }


  override val fundamentalType: TTuple = {
    val fundamentalFieldTypes = types.map(t => t.fundamentalType)
    if ((types, fundamentalFieldTypes).zipped
      .forall { case (t, ft) => t == ft })
      this
    else {
      val t = TTuple(fundamentalFieldTypes)
      t.setRequired(required).asInstanceOf[TTuple]
    }
  }
}
