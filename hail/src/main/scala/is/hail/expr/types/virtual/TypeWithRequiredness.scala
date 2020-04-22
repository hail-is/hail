package is.hail.expr.types.virtual

import is.hail.annotations.Annotation
import is.hail.expr.types.TableType
import is.hail.expr.types.physical._
import is.hail.utils.{FastSeq, Interval}
import org.apache.spark.sql.Row

object TypeWithRequiredness {
  def apply(typ: Type): TypeWithRequiredness = typ match {
    case TInt32 | TInt64 | TFloat32 | TFloat64 | TBinary | TString | TCall | TVoid | _: TLocus => RPrimitive()
    case t: TArray => RIterable(apply(t.elementType))
    case t: TSet => RIterable(apply(t.elementType))
    case t: TStream => RIterable(apply(t.elementType))
    case t: TDict => RDict(apply(t.keyType), apply(t.valueType))
    case t: TNDArray => RNDArray(apply(t.elementType))
    case t: TInterval => RInterval(apply(t.pointType), apply(t.pointType))
    case t: TStruct => RStruct(t.fields.map(f => f.name -> apply(f.typ)))
    case t: TTuple => RTuple(t.fields.map(f => apply(f.typ)))
    case t: TUnion => RUnion(t.cases.map(c => c.name -> apply(c.typ)))
  }

  def apply(t: TableType): RTable = RTable(
    t.rowType.fields.map(f => f.name -> apply(f.typ)),
    t.globalType.fields.map(f => f.name -> apply(f.typ)),
    t.key)
}


abstract class TypeWithRequiredness {
  private[this] var _required: Boolean = true
  protected[TypeWithRequiredness] var change = false

  def required: Boolean = _required

  def nested: Seq[TypeWithRequiredness]
  def _unionLiteral(a: Annotation): Unit
  def _unionPType(pType: PType): Unit

  def union(r: Boolean): Unit = {
    change = !r && required
  }

  def maximize(): Unit = {
    change = required
    nested.foreach(_.maximize())
  }

  def unionNested(req: TypeWithRequiredness): Unit =
    nested.zip(req.nested).foreach { case (r1, r2) => r1.unionFrom(r2) }

  def unionNested(reqs: Seq[TypeWithRequiredness]): Unit = reqs.foreach(unionNested)

  def unionFrom(req: TypeWithRequiredness): Unit = {
    union(req.required)
    unionNested(req)
  }

  def unionFrom(reqs: Seq[TypeWithRequiredness]): Unit = reqs.foreach(unionFrom)

  def unionLiteral(a: Annotation): Unit = {
    if (a == null)
      maximize()
    _unionLiteral(a: Annotation)
  }

  def fromPType(pType: PType): Unit = {
    union(pType.required)
    _unionPType(pType)
  }

  def probeChangedAndReset(): Boolean = {
    var hasChanged = change
    _required &= !change
    change = false
    nested.foreach { r => hasChanged |= r.probeChangedAndReset() }
    hasChanged
  }
//
//  def pretty(t: Type, compact: Boolean = true): String = {
//    val sb = new StringBuilder()
//    pretty(t, sb, if (compact) -1 else 0)
//    sb.result()
//  }
//
//  def pretty(t: Type, sb: StringBuilder, indent: Int): Unit = {
//    val compact = indent < 0
//    if (!compact)
//      sb ++= " " * indent
//    if (required)
//      sb += '+'
//    t match {
//      case t: TArray =>
//        val Seq(elt) = nested
//        sb ++= "TArray["
//        if (compact)
//          elt.pretty(t.elementType, sb, indent)
//        else {
//          sb += '\n'
//          elt.pretty(t.elementType, sb, indent + 2)
//        }
//        sb += ']'
//      case t: TStream =>
//        val Seq(elt) = nested
//        sb ++= "TStream["
//        if (compact)
//          elt.pretty(t.elementType, sb, indent)
//        else {
//          sb += '\n'
//          elt.pretty(t.elementType, sb, indent + 2)
//        }
//        sb += ']'
//      case t: TSet =>
//        val Seq(elt) = nested
//        sb ++= "TSet["
//        if (compact)
//          elt.pretty(t.elementType, sb, indent)
//        else {
//          sb += '\n'
//          elt.pretty(t.elementType, sb, indent + 2)
//        }
//        sb += ']'
//      case t: TDict =>
//        val Seq(ValueRequiredness(_, Seq(kr, vr))) = nested
//        sb ++= "TDict["
//        if (compact) {
//          vr.pretty(t.valueType, sb, indent)
//          sb += ','
//          kr.pretty(t.keyType, sb, indent)
//        } else {
//          sb += '\n'
//          vr.pretty(t.valueType, sb, indent + 2)
//          sb += '\n'
//          kr.pretty(t.keyType, sb, indent + 2)
//        }
//        sb += ']'
//      case t: TInterval =>
//        val Seq(sr, er, _, _) = nested
//        sb ++= "TInterval["
//        if (compact) {
//          sr.pretty(t.pointType, sb, indent)
//          sb += ','
//          er.pretty(t.pointType, sb, indent)
//        } else {
//          sb += '\n'
//          sr.pretty(t.pointType, sb, indent + 2)
//          sb += '\n'
//          er.pretty(t.pointType, sb, indent + 2)
//        }
//        sb += ']'
//      case t: TNDArray =>
//        val Seq(elt) = nested
//        sb ++= s"TNDArray[${ t.nDims },"
//        if (compact)
//          elt.pretty(t.elementType, sb, indent)
//        else {
//          sb += '\n'
//          elt.pretty(t.elementType, sb, indent + 2)
//        }
//        sb += ']'
//      case t: TStruct =>
//        sb ++= s"TStruct["
//        if (compact)
//          t.fields.foreachBetween { case Field(n, ft, i) =>
//            sb ++= n
//            sb += ':'
//            nested(i).pretty(ft, sb, indent)
//          }(sb += ',')
//        else
//          t.fields.foreach { case Field(n, ft, i) =>
//            sb += '\n'
//            sb ++= n
//            sb += ':'
//            nested(i).pretty(ft, sb, indent + 2)
//          }
//        sb += ']'
//      case t: TTuple =>
//        sb ++= s"TTuple["
//        if (compact)
//          t.fields.foreachBetween { case Field(n, ft, i) =>
//            nested(i).pretty(ft, sb, indent)
//          }(sb += ',')
//        else
//          t.fields.foreach { case Field(n, ft, i) =>
//            sb += '\n'
//            nested(i).pretty(ft, sb, indent + 2)
//          }
//        sb += ']'
//      case _ => sb ++= t.parsableString()
//    }
}

case class RPrimitive() extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = FastSeq.empty
  def _unionLiteral(a: Annotation): Unit = ()
  def _unionPType(pType: PType): Unit = ()
}

case class RIterable(elementType: TypeWithRequiredness) extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = FastSeq(elementType)
  def _unionLiteral(a: Annotation): Unit =
    a.asInstanceOf[Iterable[_]].foreach(elt => elementType.unionLiteral(elt))
  def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PIterable].elementType)
}
case class RDict(keyType: TypeWithRequiredness, valueType: TypeWithRequiredness)
  extends RIterable(RStruct(Array("key" -> keyType, "value" -> valueType))) {
  override def _unionLiteral(a: Annotation): Unit =
    a.asInstanceOf[Map[_,_]].foreach { case (k, v) =>
      keyType.unionLiteral(k)
      valueType.unionLiteral(v)
    }
  override def probeChangedAndReset(): Boolean = {
    elementType.change = false // TDict elements are always present, although keys/values may be missing
    super.probeChangedAndReset()
  }
}
case class RNDArray(elementType: TypeWithRequiredness) extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = FastSeq(elementType)
  def _unionLiteral(a: Annotation): Unit = ???
  def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PNDArray].elementType)
  override def probeChangedAndReset(): Boolean = {
    elementType.change = false // NDArray elements are always present and will throw a runtime error
    super.probeChangedAndReset()
  }
}

case class RInterval(startType: TypeWithRequiredness, endType: TypeWithRequiredness) extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = FastSeq(startType, endType)
  def _unionLiteral(a: Annotation): Unit = {
    startType.unionLiteral(a.asInstanceOf[Interval].start)
    endType.unionLiteral(a.asInstanceOf[Interval].end)
  }
  def _unionPType(pType: PType): Unit = {
    startType.fromPType(pType.asInstanceOf[PInterval].pointType)
    endType.fromPType(pType.asInstanceOf[PInterval].pointType)
  }
}

case class RStruct(fields: Seq[(String, TypeWithRequiredness)]) extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = fields.map(_._2)
  val fieldType: Map[String, TypeWithRequiredness] = fields.toMap
  def field(name: String): TypeWithRequiredness = fieldType(name)

  def _unionLiteral(a: Annotation): Unit =
    nested.zip(a.asInstanceOf[Row].toSeq).foreach { case (r, f) => r.unionLiteral(f) }
  def _unionPType(pType: PType): Unit =
    pType.asInstanceOf[PStruct].fields.foreach(f => nested(f.index).fromPType(f.typ))
}
case class RTuple(fields: Seq[TypeWithRequiredness]) extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = fields

  def _unionLiteral(a: Annotation): Unit =
    nested.zip(a.asInstanceOf[Row].toSeq).foreach { case (r, f) => r.unionLiteral(f) }
  def _unionPType(pType: PType): Unit =
    pType.asInstanceOf[PTuple].fields.foreach(f => nested(f.index).fromPType(f.typ))
}
case class RUnion(cases: Seq[(String, TypeWithRequiredness)]) extends TypeWithRequiredness {
  val nested: Seq[TypeWithRequiredness] = cases.map(_._2)
  def _unionLiteral(a: Annotation): Unit = ???
  def _unionPType(pType: PType): Unit = ???
}

case class RTable(rowFields: Seq[(String, TypeWithRequiredness)], globalFields: Seq[(String, TypeWithRequiredness)], key: Seq[String]) extends TypeWithRequiredness {
  val rowTypes: Seq[TypeWithRequiredness] = rowFields.map(_._2)
  val globalTypes: Seq[TypeWithRequiredness] = globalFields.map(_._2)

  val fieldMap: Map[String, TypeWithRequiredness] = (rowFields ++ globalFields).toMap
  def field(name: String): TypeWithRequiredness = fieldMap(name)

  val nested: Seq[TypeWithRequiredness] = rowTypes ++ globalTypes
  def _unionLiteral(a: Annotation): Unit = ???
  def _unionPType(pType: PType): Unit = ???

  val rowRequired: TypeWithRequiredness = RStruct(rowFields)
  val globalRequired: TypeWithRequiredness = RStruct(globalFields)

  def unionRows(req: TypeWithRequiredness): Unit = req match {
    case r: RTable => rowFields.zip(r.rowFields).foreach { case ((_, r1), (_, r2)) => r1.unionFrom(r2) }
    case r: RStruct => rowFields.zip(r.nested).foreach { case ((_, r1), r2) => r1.unionFrom(r2) }
  }

  def unionGlobals(req: TypeWithRequiredness): Unit = req match {
    case r: RTable => globalFields.zip(r.globalFields).foreach { case ((_, r1), (_, r2)) => r1.unionFrom(r2) }
    case r: RStruct => globalFields.zip(r.nested).foreach { case ((_, r1), r2) => r1.unionFrom(r2) }
  }
}