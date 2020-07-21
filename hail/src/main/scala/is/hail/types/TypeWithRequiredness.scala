package is.hail.types

import is.hail.annotations.Annotation
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval}
import org.apache.spark.sql.Row

object BaseTypeWithRequiredness {
  def apply(typ: BaseType): BaseTypeWithRequiredness = typ match {
    case t: Type => TypeWithRequiredness(t)
    case t: TableType => RTable(
      t.rowType.fields.map(f => f.name -> TypeWithRequiredness(f.typ)),
      t.globalType.fields.map(f => f.name -> TypeWithRequiredness(f.typ)),
      t.key)
  }

  def check(r: BaseTypeWithRequiredness, typ: BaseType): Unit = {
    r match {
      case r: RTable =>
        val table = typ.asInstanceOf[TableType]
        check(r.globalType, table.globalType)
        check(r.rowType, table.rowType)
        assert(r.key == table.key)
      case _: RPrimitive => assert(RPrimitive.typeSupported(typ.asInstanceOf[Type]))
      case r: RIterable =>
        check(r.elementType, typ.asInstanceOf[TIterable])
      case r: RNDArray =>
        check(r.elementType, typ.asInstanceOf[TNDArray].elementType)
      case r: RInterval =>
        check(r.startType, typ.asInstanceOf[TInterval].pointType)
        check(r.endType, typ.asInstanceOf[TInterval].pointType)
      case r: RStruct =>
        val struct = typ.asInstanceOf[TStruct]
        (r.fields, struct.fields).zipped.map { case (rf, f) =>
          assert(rf.name == f.name)
          check(rf.typ, f.typ)
        }
      case r: RTuple =>
        val tuple = typ.asInstanceOf[TTuple]
        (r.fields, tuple.fields).zipped.map { case (rf, f) =>
          assert(rf.index == f.index)
          check(rf.typ, f.typ)
        }
      case r: RUnion =>
        val union = typ.asInstanceOf[TUnion]
        (r.cases, union.cases).zipped.map { case (rc, c) =>
          assert(rc._1 == c.name)
          check(rc._2, c.typ)
        }
    }
  }
}

object TypeWithRequiredness {
  def apply(typ: Type): TypeWithRequiredness = typ match {
    case t if RPrimitive.typeSupported(t) => RPrimitive()
    case t: TArray => RIterable(apply(t.elementType))
    case t: TSet => RIterable(apply(t.elementType))
    case t: TStream => RIterable(apply(t.elementType))
    case t: TDict => RDict(apply(t.keyType), apply(t.valueType))
    case t: TNDArray => RNDArray(apply(t.elementType))
    case t: TInterval => RInterval(apply(t.pointType), apply(t.pointType))
    case t: TStruct => RStruct(t.fields.map(f => f.name -> apply(f.typ)))
    case t: TTuple => RTuple(t.fields.map(f => RField(f.name, apply(f.typ), f.index)))
    case t: TUnion => RUnion(t.cases.map(c => c.name -> apply(c.typ)))
  }
}

sealed abstract class BaseTypeWithRequiredness {
  private[this] var _required: Boolean = true
  private[this] var change = false

  def required: Boolean = _required & !change
  def children: Seq[BaseTypeWithRequiredness]
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): BaseTypeWithRequiredness
  def toString: String

  def minimalCopy(): BaseTypeWithRequiredness =
    copy(children.map(_.minimalCopy()))

  def deepCopy(): BaseTypeWithRequiredness = {
    val r = minimalCopy()
    r.unionFrom(this)
    r
  }

  protected[this] def _maximizeChildren(): Unit = children.foreach(_.maximize())
  protected[this] def _unionChildren(newChildren: Seq[BaseTypeWithRequiredness]): Unit = {
    if (children.length != newChildren.length) {
      throw new AssertionError(
        s"children lengths differed ${children.length} ${newChildren.length}. ${children} ${newChildren} ${this}")
    }
    children.zip(newChildren).foreach { case (r1, r2) =>
      r1.unionFrom(r2)
    }
  }

  protected[this] def _unionWithIntersection(ts: Seq[BaseTypeWithRequiredness]): Unit = {
    var i = 0
    while(i < children.length) {
      children(i).unionWithIntersection(ts.map(_.children(i)))
      i += 1
    }
  }
  def unionWithIntersection(ts: Seq[BaseTypeWithRequiredness]): Unit = {
    union(ts.exists(_.required))
    _unionWithIntersection(ts)
  }

  final def union(r: Boolean): Unit = { change |= !r && required }
  final def maximize(): Unit = {
    change |= required
    _maximizeChildren()
  }

  final def unionFrom(req: BaseTypeWithRequiredness): Unit = {
    union(req.required)
    _unionChildren(req.children)
  }

  final def unionFrom(reqs: Seq[BaseTypeWithRequiredness]): Unit = reqs.foreach(unionFrom)

  final def probeChangedAndReset(): Boolean = {
    var hasChanged = change
    _required &= !change
    change = false
    children.foreach { r => hasChanged |= r.probeChangedAndReset() }
    hasChanged
  }
}

sealed abstract class TypeWithRequiredness extends BaseTypeWithRequiredness {
  def _unionLiteral(a: Annotation): Unit
  def _unionPType(pType: PType): Unit
  def _matchesPType(pt: PType): Boolean
  def unionLiteral(a: Annotation): Unit =
    if (a == null) union(false) else _unionLiteral(a)

  def fromPType(pType: PType): Unit = {
    union(pType.required)
    _unionPType(pType)
  }
  def canonicalPType(t: Type): PType
  def matchesPType(pt: PType): Boolean = pt.required == required && _matchesPType(pt)
  def _toString: String
  override def toString: String = if (required) "+" + _toString else _toString
}

object RPrimitive {
  val supportedTypes: Set[Type] = Set(TBoolean, TInt32, TInt64, TFloat32, TFloat64, TBinary, TString, TCall, TVoid)
  def typeSupported(t: Type): Boolean = RPrimitive.supportedTypes.contains(t) ||
    t.isInstanceOf[TLocus] ||
    t.isInstanceOf[TShuffle]
}

final case class RPrimitive() extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq.empty

  def _unionLiteral(a: Annotation): Unit = ()
  def _matchesPType(pt: PType): Boolean = RPrimitive.typeSupported(pt.virtualType)
  def _unionPType(pType: PType): Unit = assert(RPrimitive.typeSupported(pType.virtualType))
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RPrimitive = {
    assert(newChildren.isEmpty)
    RPrimitive()
  }
  def canonicalPType(t: Type): PType = {
    assert(RPrimitive.typeSupported(t))
    PType.canonical(t, required)
  }
  def _toString: String = "RPrimitive"
}

object RIterable {
  def apply(elementType: TypeWithRequiredness): RIterable = new RIterable(elementType, eltRequired = false)
  def unapply(r: RIterable): Option[TypeWithRequiredness] = Some(r.elementType)
}

sealed class RIterable(val elementType: TypeWithRequiredness, eltRequired: Boolean) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq(elementType)
  def _unionLiteral(a: Annotation): Unit =
    a.asInstanceOf[Iterable[_]].foreach(elt => elementType.unionLiteral(elt))
  def _matchesPType(pt: PType): Boolean = elementType.matchesPType(coerce[PIterable](pt).elementType)
  def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PIterable].elementType)
  def _toString: String = s"RIterable[${ elementType.toString }]"

  override def _maximizeChildren(): Unit = {
    if (eltRequired)
      elementType.children.foreach(_.maximize())
    else elementType.maximize()
  }

  override def _unionChildren(newChildren: Seq[BaseTypeWithRequiredness]): Unit = {
    val Seq(newEltReq) = newChildren
    unionElement(newEltReq)
  }

  override def _unionWithIntersection(ts: Seq[BaseTypeWithRequiredness]): Unit = {
    if (eltRequired) {
      var i = 0
      while(i < elementType.children.length) {
        elementType.children(i).unionWithIntersection(ts.map(t => coerce[RIterable](t).elementType.children(i)))
        i += 1
      }
    } else
      elementType.unionWithIntersection(ts.map(t => coerce[RIterable](t).elementType))
  }

  def unionElement(newElement: BaseTypeWithRequiredness): Unit = {
    if (eltRequired)
      elementType.children.zip(newElement.children).foreach { case (r1, r2) => r1.unionFrom(r2) }
    else
      elementType.unionFrom(newElement)
  }

  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RIterable = {
    val Seq(newElt: TypeWithRequiredness) = newChildren
    RIterable(newElt)
  }
  def canonicalPType(t: Type): PType = {
    val elt = elementType.canonicalPType(coerce[TIterable](t).elementType)
    t match {
      case _: TArray => PCanonicalArray(elt, required = required)
      case _: TSet => PCanonicalSet(elt, required = required)
      case _: TStream => PCanonicalStream(elt, required = required)
    }
  }
}
case class RDict(keyType: TypeWithRequiredness, valueType: TypeWithRequiredness)
  extends RIterable(RStruct(Array("key" -> keyType, "value" -> valueType)), true) {
  override def _unionLiteral(a: Annotation): Unit =
    a.asInstanceOf[Map[_,_]].foreach { case (k, v) =>
      keyType.unionLiteral(k)
      valueType.unionLiteral(v)
    }
  override def copy(newChildren: Seq[BaseTypeWithRequiredness]): RDict = {
    val Seq(newElt: RStruct) = newChildren
    RDict(newElt.field("key"), newElt.field("value"))
  }
  override def canonicalPType(t: Type): PType =
    PCanonicalDict(
      keyType.canonicalPType(coerce[TDict](t).keyType),
      valueType.canonicalPType(coerce[TDict](t).valueType),
      required = required)
  override def _toString: String = s"RDict[${ keyType.toString }, ${ valueType.toString }]"
}
case class RNDArray(override val elementType: TypeWithRequiredness) extends RIterable(elementType, true) {
  override def _unionLiteral(a: Annotation): Unit = {
    val data = a.asInstanceOf[Row].getAs[Iterable[Any]](2)
    data.asInstanceOf[Iterable[_]].foreach { elt =>
      if (elt != null)
        elementType.unionLiteral(elt)
    }
  }
  override def _matchesPType(pt: PType): Boolean = elementType.matchesPType(coerce[PNDArray](pt).elementType)
  override def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PNDArray].elementType)
  override def copy(newChildren: Seq[BaseTypeWithRequiredness]): RNDArray = {
    val Seq(newElt: TypeWithRequiredness) = newChildren
    RNDArray(newElt)
  }
  override def canonicalPType(t: Type): PType = {
    val tnd = coerce[TNDArray](t)
    PCanonicalNDArray(elementType.canonicalPType(tnd.elementType), tnd.nDims, required = required)
  }
  override def _toString: String = s"RNDArray[${ elementType.toString }]"
}

case class RInterval(startType: TypeWithRequiredness, endType: TypeWithRequiredness) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = FastSeq(startType, endType)
  def _unionLiteral(a: Annotation): Unit = {
    startType.unionLiteral(a.asInstanceOf[Interval].start)
    endType.unionLiteral(a.asInstanceOf[Interval].end)
  }
  def _matchesPType(pt: PType): Boolean =
    startType.matchesPType(coerce[PInterval](pt).pointType) &&
      endType.matchesPType(coerce[PInterval](pt).pointType)
  def _unionPType(pType: PType): Unit = {
    startType.fromPType(pType.asInstanceOf[PInterval].pointType)
    endType.fromPType(pType.asInstanceOf[PInterval].pointType)
  }
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RInterval = {
    val Seq(newStart: TypeWithRequiredness, newEnd: TypeWithRequiredness) = newChildren
    RInterval(newStart, newEnd)
  }

  def canonicalPType(t: Type): PType = t match {
    case TInterval(pointType) =>
      val unified = startType.deepCopy().asInstanceOf[TypeWithRequiredness]
      unified.unionFrom(endType)
      PCanonicalInterval(unified.canonicalPType(pointType), required = required)
  }
  def _toString: String = s"RInterval[${ startType.toString }, ${ endType.toString }]"
}


case class RField(name: String, typ: TypeWithRequiredness, index: Int)
sealed abstract class RBaseStruct extends TypeWithRequiredness {
  def fields: IndexedSeq[RField]
  val children: Seq[TypeWithRequiredness] = fields.map(_.typ)
  def _unionLiteral(a: Annotation): Unit =
    children.zip(a.asInstanceOf[Row].toSeq).foreach { case (r, f) => r.unionLiteral(f) }
  def _matchesPType(pt: PType): Boolean =
    coerce[PBaseStruct](pt).fields.forall(f => children(f.index).matchesPType(f.typ))
  def _unionPType(pType: PType): Unit =
    pType.asInstanceOf[PBaseStruct].fields.foreach(f => children(f.index).fromPType(f.typ))
  def canonicalPType(t: Type): PType = t match {
    case ts: TStruct =>
      PCanonicalStruct(required = required,
        fields.map(f => f.name -> f.typ.canonicalPType(ts.fieldType(f.name))): _*)
    case ts: TTuple =>
      PCanonicalTuple(fields.zip(ts.fields).map { case(fr, ft) =>
        assert(fr.index == ft.index)
        PTupleField(fr.index, fr.typ.canonicalPType(ft.typ))
      }, required = required)
  }
}

object RStruct {
  def apply(fields: Seq[(String, TypeWithRequiredness)]): RStruct =
    RStruct(Array.tabulate(fields.length)(i => RField(fields(i)._1, fields(i)._2, i)))
}
case class RStruct(fields: IndexedSeq[RField]) extends RBaseStruct {
  val fieldType: Map[String, TypeWithRequiredness] = fields.map(f => f.name -> f.typ).toMap
  def field(name: String): TypeWithRequiredness = fieldType(name)
  def hasField(name: String): Boolean = fieldType.contains(name)
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RStruct = {
    assert(newChildren.length == fields.length)
    RStruct(Array.tabulate(fields.length)(i => fields(i).name -> coerce[TypeWithRequiredness](newChildren(i))))
  }
  def _toString: String = s"RStruct[${ fields.map(f => s"${ f.name }: ${ f.typ.toString }").mkString(",") }]"
}

case class RTuple(fields: IndexedSeq[RField]) extends RBaseStruct {
  val fieldType: Map[Int, TypeWithRequiredness] = fields.map(f => f.index -> f.typ).toMap
  def field(idx: Int): TypeWithRequiredness = fieldType(idx)
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RTuple = {
    assert(newChildren.length == fields.length)
    RTuple(fields.zip(newChildren).map { case (f, c) => RField(f.name, coerce[TypeWithRequiredness](c), f.index) })
  }
  def _toString: String = s"RTuple[${ fields.map(f => s"${ f.index }: ${ f.typ.toString }").mkString(",") }]"
}

case class RUnion(cases: Seq[(String, TypeWithRequiredness)]) extends TypeWithRequiredness {
  val children: Seq[TypeWithRequiredness] = cases.map(_._2)
  def _unionLiteral(a: Annotation): Unit = ???
  def _matchesPType(pt: PType): Boolean = ???
  def _unionPType(pType: PType): Unit = ???
  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RUnion = {
    assert(newChildren.length == cases.length)
    RUnion(Array.tabulate(cases.length)(i => cases(i)._1 -> coerce[TypeWithRequiredness](newChildren(i))))
  }
  def canonicalPType(t: Type): PType = ???
  def _toString: String = s"RStruct[${ cases.map { case (n, t) => s"${ n }: ${ t.toString }" }.mkString(",") }]"
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
    assert(key.length <= req.key.length)
    key.zip(req.key).foreach { case (k, rk) => field(k).unionFrom(req.field(rk)) }
  }

  def unionValues(req: RStruct): Unit = valueFields.foreach { n => if (req.hasField(n)) field(n).unionFrom(req.field(n)) }
  def unionValues(req: RTable): Unit = unionValues(req.rowType)

  def changeKey(key: Seq[String]): RTable = RTable(rowFields, globalFields, key)

  def copy(newChildren: Seq[BaseTypeWithRequiredness]): RTable = {
    assert(newChildren.length == rowFields.length + globalFields.length)
    val newRowFields = rowFields.zip(newChildren.take(rowFields.length)).map { case ((n, _), r: TypeWithRequiredness) => n -> r }
    val newGlobalFields = globalFields.zip(newChildren.drop(rowFields.length)).map { case ((n, _), r: TypeWithRequiredness) => n -> r }
    RTable(newRowFields, newGlobalFields, key)
  }

  def asMatrixType(colField: String, entryField: String): RMatrix = {
    val row = RStruct(rowFields.filter(_._1 != entryField))
    val entry = coerce[RStruct](coerce[RIterable](field(entryField)).elementType)
    val col = coerce[RStruct](coerce[RIterable](field(colField)).elementType)
    val global = RStruct(globalFields.filter(_._1 != colField))
    RMatrix(row, entry, col, global)
  }

  override def toString: String =
    s"RTable[\n  row:${ rowType.toString }\n  global:${ globalType.toString }]"
}

case class RMatrix(rowType: RStruct, entryType: RStruct, colType: RStruct, globalType: RStruct) {
  val entriesRVType: RStruct = RStruct(Seq(MatrixType.entriesIdentifier -> RIterable(entryType)))
}
