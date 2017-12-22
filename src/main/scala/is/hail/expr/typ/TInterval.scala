package is.hail.expr.typ

/**
  * Created by dking on 12/21/17.
  */

case class TInterval(pointType: Type, override val required: Boolean = false) extends ComplexType {
  override def children = Seq(pointType)

  def _toString = s"""Interval[$pointType]"""

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Interval[")
    pointType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def _typeCheck(a: Any): Boolean = a.isInstanceOf[Interval[_]] && {
    val i = a.asInstanceOf[Interval[_]]
    pointType.typeCheck(i.start) && pointType.typeCheck(i.end)
  }

  override def genNonmissingValue: Gen[Annotation] = Interval.gen(pointType.genValue)(pointType.ordering(true))

  override def desc: String = "An ``Interval[T]`` is a Hail data type representing a range over ordered values of type T."

  override def scalaClassTag: ClassTag[Interval[Annotation]] = classTag[Interval[Annotation]]

  override def ordering(missingGreatest: Boolean): Ordering[Annotation] = {
    annotationOrdering(
      extendOrderingToNull(missingGreatest)(Interval.ordering[Annotation]))
  }

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = representation.unsafeOrdering(missingGreatest)

  val representation: TStruct = {
    val rep = TStruct(
      "start" -> pointType,
      "end" -> pointType)
    if (required) (!rep).asInstanceOf[TStruct] else rep
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TInterval(cpointType, _) => pointType.unify(cpointType)
    case _ => false
  }

  override def subst() = TInterval(pointType.subst())
}
