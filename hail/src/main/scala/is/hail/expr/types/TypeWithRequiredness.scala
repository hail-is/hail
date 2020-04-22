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
  def children: Seq[BaseTypeWithRequiredness]
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): BaseTypeWithRequiredness

  def union(r: Boolean): Unit = { change = !r && required }
  def maximize(): Unit = {
    change = required
    children.foreach(_.maximize())
  }

  def unionFrom(req: BaseTypeWithRequiredness): Unit = {
    union(req.required)
    children.zip(req.children).foreach { case (r1, r2) => r1.unionFrom(r2) }
  }

  def unionFrom(reqs: Seq[BaseTypeWithRequiredness]): Unit = reqs.foreach(unionFrom)

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
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RPrimitive = {
    assert(newChildren.isEmpty)
    RPrimitive()
  }
}

object RIterable {
  def apply(elementType: TypeWithRequiredness): RIterable = new RIterable(elementType)
}

class RIterable(val elementType: TypeWithRequiredness) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq(elementType)
  def _unionLiteral(a: Annotation): Unit =
    a.asInstanceOf[Iterable[_]].foreach(elt => elementType.unionLiteral(elt))
  def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PIterable].elementType)
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RIterable = {
    val Seq(newElt: TypeWithRequiredness) = newChildren
    RIterable(newElt)
  }
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
  override def copy(newChildren: Seq[BaseTypeWithRequiredness]): RDict = {
    val Seq(newElt: RStruct) = newChildren
    RDict(newElt.field("key"), newElt.field("value"))
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
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RNDArray = {
    val Seq(newElt: TypeWithRequiredness) = newChildren
    RNDArray(newElt)
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
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RInterval = {
    val Seq(newStart: TypeWithRequiredness, newEnd: TypeWithRequiredness) = newChildren
    RInterval(newStart, newEnd)
  }
}


case class RField(name: String, typ: TypeWithRequiredness, index: Int)
abstract class RBaseStruct(val fields: Seq[RField]) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = fields.map(_.typ)
  def _unionLiteral(a: Annotation): Unit =
    children.zip(a.asInstanceOf[Row].toSeq).foreach { case (r, f) => r.unionLiteral(f) }
  def _unionPType(pType: PType): Unit =
    pType.asInstanceOf[PBaseStruct].fields.foreach(f => children(f.index).fromPType(f.typ))
}

object RStruct {
  def apply(fields: Seq[(String, TypeWithRequiredness)]): RStruct =
    RStruct(Array.tabulate(fields.length)(i => RField(fields(i)._1, fields(i)._2, i)))
}
case class RStruct(override val fields: Seq[RField]) extends RBaseStruct(fields) {
  val fieldType: Map[String, TypeWithRequiredness] = fields.map(f => f.name -> f.typ).toMap
  def field(name: String): TypeWithRequiredness = fieldType(name)
  def hasField(name: String): Boolean = fieldType.contains(name)
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RStruct = {
    assert(newChildren.length == fields.length)
    RStruct(Array.tabulate(fields.length)(i => fields(i).name -> coerce[TypeWithRequiredness](newChildren(i))))
  }
}

object RTuple {
  def apply(fields: Seq[TypeWithRequiredness]): RTuple =
    RTuple(Array.tabulate(fields.length)(i => RField(s"$i", fields(i), i)))
}

case class RTuple(override val fields: Seq[RField]) extends RBaseStruct(fields) {
  val types: Seq[TypeWithRequiredness] = fields.map(_.typ)
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RTuple = {
    assert(newChildren.length == fields.length)
    RTuple(newChildren.map(coerce[TypeWithRequiredness]))
  }
}

case class RUnion(cases: Seq[(String, TypeWithRequiredness)]) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = cases.map(_._2)
  def _unionLiteral(a: Annotation): Unit = ???
  def _unionPType(pType: PType): Unit = ???
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RUnion = {
    assert(newChildren.length == cases.length)
    RUnion(Array.tabulate(cases.length)(i => cases(i)._1 -> coerce[TypeWithRequiredness](newChildren(i))))
  }
}

case class RTable(rowFields: Seq[(String, TypeWithRequiredness)], globalFields: Seq[(String, TypeWithRequiredness)], key: Seq[String]) extends BaseTypeWithRequiredness {
  val rowTypes: Seq[TypeWithRequiredness] = rowFields.map(_._2)
  val globalTypes: Seq[TypeWithRequiredness] = globalFields.map(_._2)
  val keyFields: Set[String] = key.toSet
  val valueFields: Set[String] = rowFields.map(_._1).filter(n => !keyFields.contains(n)).toSet

  val fieldMap: Map[String, TypeWithRequiredness] = (rowFields ++ globalFields).toMap
  def field(name: String): TypeWithRequiredness = fieldMap(name)

  val children: Seq[TypeWithRequiredness] = rowTypes ++ globalTypes

  val rowType: RStruct = RStruct(rowFields)
  val globalType: RStruct = RStruct(globalFields)

  def unionRows(req: RStruct): Unit = rowFields.foreach { case (n, r) => if (req.hasField(n)) r.unionFrom(req.field(n)) }
  def unionRows(req: RTable): Unit = unionRows(req.rowType)

  def unionGlobals(req: RStruct): Unit = globalFields.foreach { case (n, r) => if (req.hasField(n)) r.unionFrom(req.field(n)) }
  def unionGlobals(req: RTable): Unit = unionGlobals(req.globalType)

  def unionKeys(req: RStruct): Unit = key.foreach { n => field(n).unionFrom(req.field(n)) }
  def unionKeys(req: RTable): Unit = {
    assert(key.zip(req.key).forall { case (k1, k2) => k1 == k2 } && key.length <= req.key.length)
    unionKeys(req.rowType)
  }

  def unionValues(req: RStruct): Unit = valueFields.foreach { n => if (req.hasField(n)) field(n).unionFrom(req.field(n)) }
  def unionValues(req: RTable): Unit = unionValues(req.rowType)

  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RTable = {
    assert(newChildren.length == rowFields.length + globalFields.length)
    val newRowFields = rowFields.zip(newChildren.take(rowFields.length)).map { case ((n, _), r: TypeWithRequiredness) => n -> r }
    val newGlobalFields = globalFields.zip(newChildren.drop(rowFields.length)).map { case ((n, _), r: TypeWithRequiredness) => n -> r }
    RTable(newRowFields, newGlobalFields, key)
  }
}