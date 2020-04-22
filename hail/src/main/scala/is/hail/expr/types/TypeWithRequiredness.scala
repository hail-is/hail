package is.hail.expr.types

import is.hail.annotations.Annotation
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
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


abstract class BaseTypeWithRequiredness {
  private[this] var _required: Boolean = true
  protected[TypeWithRequiredness] var change = false

  def required: Boolean = _required
  def children: Seq[TypeWithRequiredness]

  def union(r: Boolean): Unit = { change = !r && required }
  def maximize(): Unit = {
    change = required
    children.foreach(_.maximize())
  }

  def unionChildren(req: TypeWithRequiredness): Unit =
    children.zip(req.children).foreach { case (r1, r2) => r1.unionFrom(r2) }

  def unionChildren(reqs: Seq[TypeWithRequiredness]): Unit = reqs.foreach(unionChildren)

  def unionFrom(req: TypeWithRequiredness): Unit = {
    union(req.required)
    unionChildren(req)
  }

  def unionFrom(reqs: Seq[TypeWithRequiredness]): Unit = reqs.foreach(unionFrom)

  def probeChangedAndReset(): Boolean = {
    var hasChanged = change
    _required &= !change
    change = false
    children.foreach { r => hasChanged |= r.probeChangedAndReset() }
    hasChanged
  }
}

abstract class TypeWithRequiredness extends BaseTypeWithRequiredness {
  def _unionLiteral(a: Annotation): Unit
  def _unionPType(pType: PType): Unit

  def unionLiteral(a: Annotation): Unit = {
    if (a == null)
      maximize()
    _unionLiteral(a: Annotation)
  }

  def fromPType(pType: PType): Unit = {
    union(pType.required)
    _unionPType(pType)
  }
}

case class RPrimitive() extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq.empty
  def _unionLiteral(a: Annotation): Unit = ()
  def _unionPType(pType: PType): Unit = ()
}

case class RIterable(elementType: TypeWithRequiredness) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq(elementType)
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
  val children: Seq[TypeWithRequiredness] = FastSeq(elementType)
  def _unionLiteral(a: Annotation): Unit = ???
  def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PNDArray].elementType)
  override def probeChangedAndReset(): Boolean = {
    elementType.change = false // NDArray elements are always present and will throw a runtime error
    super.probeChangedAndReset()
  }
}

case class RInterval(startType: TypeWithRequiredness, endType: TypeWithRequiredness) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq(startType, endType)
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
  val children: Seq[TypeWithRequiredness] = fields.map(_._2)
  val fieldType: Map[String, TypeWithRequiredness] = fields.toMap
  def field(name: String): TypeWithRequiredness = fieldType(name)

  def _unionLiteral(a: Annotation): Unit =
    children.zip(a.asInstanceOf[Row].toSeq).foreach { case (r, f) => r.unionLiteral(f) }
  def _unionPType(pType: PType): Unit =
    pType.asInstanceOf[PStruct].fields.foreach(f => children(f.index).fromPType(f.typ))
}
case class RTuple(fields: Seq[TypeWithRequiredness]) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = fields

  def _unionLiteral(a: Annotation): Unit =
    children.zip(a.asInstanceOf[Row].toSeq).foreach { case (r, f) => r.unionLiteral(f) }
  def _unionPType(pType: PType): Unit =
    pType.asInstanceOf[PTuple].fields.foreach(f => children(f.index).fromPType(f.typ))
}
case class RUnion(cases: Seq[(String, TypeWithRequiredness)]) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = cases.map(_._2)
  def _unionLiteral(a: Annotation): Unit = ???
  def _unionPType(pType: PType): Unit = ???
}

case class RTable(rowFields: Seq[(String, TypeWithRequiredness)], globalFields: Seq[(String, TypeWithRequiredness)], key: Seq[String]) extends BaseTypeWithRequiredness {
  val rowTypes: Seq[TypeWithRequiredness] = rowFields.map(_._2)
  val globalTypes: Seq[TypeWithRequiredness] = globalFields.map(_._2)

  val fieldMap: Map[String, TypeWithRequiredness] = (rowFields ++ globalFields).toMap
  def field(name: String): TypeWithRequiredness = fieldMap(name)

  val children: Seq[TypeWithRequiredness] = rowTypes ++ globalTypes

  val rowRequired: TypeWithRequiredness = RStruct(rowFields)
  val globalRequired: TypeWithRequiredness = RStruct(globalFields)

  def unionRows(req: BaseTypeWithRequiredness): Unit = req match {
    case r: RTable => rowFields.zip(r.rowFields).foreach { case ((_, r1), (_, r2)) => r1.unionFrom(r2) }
    case r: RStruct => rowFields.zip(r.children).foreach { case ((_, r1), r2) => r1.unionFrom(r2) }
  }

  def unionGlobals(req: BaseTypeWithRequiredness): Unit = req match {
    case r: RTable => globalFields.zip(r.globalFields).foreach { case ((_, r1), (_, r2)) => r1.unionFrom(r2) }
    case r: RStruct => globalFields.zip(r.children).foreach { case ((_, r1), r2) => r1.unionFrom(r2) }
  }
}