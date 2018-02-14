package is.hail.expr.types

import is.hail.annotations.ExtendedOrdering
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

final case class TTuple(types: IndexedSeq[Type], override val required: Boolean = false) extends TStructBase {

  val ordering: ExtendedOrdering =
    ExtendedOrdering.rowOrdering(types.map(_.ordering).toArray)

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
