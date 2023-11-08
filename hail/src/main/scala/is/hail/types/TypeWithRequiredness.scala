package is.hail.types

import is.hail.annotations.{Annotation, NDArray}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.ir.{ComputeUsesAndDefs, Env, IR}
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SInterval, SNDArray, SStream}
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval, toMapFast}
import org.apache.spark.sql.Row

object BaseTypeWithRequiredness {
  def apply(typ: BaseType): BaseTypeWithRequiredness = typ match {
    case t: Type => TypeWithRequiredness(t)
    case t: TableType => RTable(
      t.rowType.fields.map(f => f.name -> TypeWithRequiredness(f.typ)),
      t.globalType.fields.map(f => f.name -> TypeWithRequiredness(f.typ)),
      t.key)
    case t: BlockMatrixType => RBlockMatrix(TypeWithRequiredness(t.elementType))
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
        (r.fields, struct.fields).zipped.map { (rf, f) =>
          assert(rf.name == f.name)
          check(rf.typ, f.typ)
        }
      case r: RTuple =>
        val tuple = typ.asInstanceOf[TTuple]
        (r.fields, tuple.fields).zipped.map { (rf, f) =>
          assert(rf.index == f.index)
          check(rf.typ, f.typ)
        }
      case r: RUnion =>
        val union = typ.asInstanceOf[TUnion]
        (r.cases, union.cases).zipped.map { (rc, c) =>
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
    case t: TStruct => RStruct.fromNamesAndTypes(t.fields.map(f => f.name -> apply(f.typ)))
    case t: TTuple => RTuple.fromNamesAndTypes(t.fields.map(f => f.name -> apply(f.typ)))
    case t: TUnion => RUnion(t.cases.map(c => c.name -> apply(c.typ)))
  }
}

sealed abstract class BaseTypeWithRequiredness {
  private[this] var _required: Boolean = true
  private[this] var change = false

  def required: Boolean = _required & !change
  def children: IndexedSeq[BaseTypeWithRequiredness]
  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): BaseTypeWithRequiredness
  def toString: String

  def minimalCopy(): BaseTypeWithRequiredness =
    copy(children.map(_.minimalCopy()))

  def deepCopy(): BaseTypeWithRequiredness = {
    val r = minimalCopy()
    r.unionFrom(this)
    r
  }

  protected[this] def _maximizeChildren(): Unit = children.foreach(_.maximize())
  protected[this] def _unionChildren(newChildren: IndexedSeq[BaseTypeWithRequiredness]): Unit = {
    if (children.length != newChildren.length) {
      throw new AssertionError(
        s"children lengths differed ${children.length} ${newChildren.length}. ${children} ${newChildren} ${this}")
    }

    // foreach on zipped seqs is very slow as the implementation
    // doesn't know that the seqs are the same length.
    for (i <- children.indices) {
      children(i).unionFrom(newChildren(i))
    }
  }

  protected[this] def _unionWithIntersection(ts: IndexedSeq[BaseTypeWithRequiredness]): Unit = {
    var i = 0
    while(i < children.length) {
      children(i).unionWithIntersection(ts.map(_.children(i)))
      i += 1
    }
  }
  def unionWithIntersection(ts: IndexedSeq[BaseTypeWithRequiredness]): Unit = {
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

  final def unionFrom(reqs: IndexedSeq[BaseTypeWithRequiredness]): Unit = reqs.foreach(unionFrom)

  final def probeChangedAndReset(): Boolean = {
    var hasChanged = change
    _required &= !change
    change = false
    children.foreach { r => hasChanged |= r.probeChangedAndReset() }
    hasChanged
  }

  def hardSetRequiredness(newRequiredness: Boolean): Unit = {
    _required = newRequiredness
    change = false
  }
}

object VirtualTypeWithReq {
  def apply(pt: PType): VirtualTypeWithReq = {
    val vt = pt.virtualType
    val r = TypeWithRequiredness(vt)
    r.fromPType(pt)
    VirtualTypeWithReq(vt, r)
  }

  def fullyOptional(t: Type): VirtualTypeWithReq = {
    val twr = TypeWithRequiredness(t)
    twr.fromPType(PType.canonical(t))
    assert(!twr.required)
    VirtualTypeWithReq(t, twr)
  }

  def fullyRequired(t: Type): VirtualTypeWithReq = {
    val twr = TypeWithRequiredness(t)
    assert(twr.required)
    VirtualTypeWithReq(t, twr)
  }

  def fromLiteral(t: Type, value: Annotation): VirtualTypeWithReq = {
    val twr = TypeWithRequiredness(t)
    twr.unionLiteral(value)
    VirtualTypeWithReq(t, twr)
}
  def union(vs: IndexedSeq[VirtualTypeWithReq]): VirtualTypeWithReq = {
    val t = vs.head.t
    assert(vs.tail.forall(_.t == t))

    val tr = TypeWithRequiredness(t)
    tr.unionFrom(vs.map(_.r))
    VirtualTypeWithReq(t, tr)
  }

  def subset(vt: Type, rt: TypeWithRequiredness): VirtualTypeWithReq = {
    val empty = FastSeq()
    def subsetRT(vt: Type, rt: TypeWithRequiredness): TypeWithRequiredness = {
      val r = (vt, rt) match {
        case (_, t: RPrimitive) => t.copy(empty)
        case (tt: TTuple, rt: RTuple) =>
          RTuple(tt.fields.map(fd => RField(fd.name, subsetRT(fd.typ, rt.fieldType(fd.name)), fd.index)))
        case (ts: TStruct, rt: RStruct) =>
          RStruct(ts.fields.map(fd => RField(fd.name, subsetRT(fd.typ, rt.field(fd.name)), fd.index)))
        case (ti: TInterval, ri: RInterval) => RInterval(subsetRT(ti.pointType, ri.startType), subsetRT(ti.pointType, ri.endType))
        case (td: TDict, ri: RDict) => RDict(subsetRT(td.keyType, ri.keyType), subsetRT(td.valueType, ri.valueType))
        case (tit: TIterable, rit: RIterable) => RIterable(subsetRT(tit.elementType, rit.elementType))
        case (tnd: TNDArray, rnd: RNDArray) => RNDArray(subsetRT(tnd.elementType, rnd.elementType))
      }
      r.union(rt.required)
      r
    }
    VirtualTypeWithReq(vt, subsetRT(vt, rt))
  }
}

case class VirtualTypeWithReq(t: Type, r: TypeWithRequiredness) {
  lazy val canonicalPType: PType = r.canonicalPType(t)
  lazy val canonicalEmitType: EmitType = {
    t match {
      case ts: TStream =>
        EmitType(SStream(VirtualTypeWithReq(ts.elementType, r.asInstanceOf[RIterable].elementType).canonicalEmitType), r.required)
      case t =>
        val pt = r.canonicalPType(t)
        EmitType(pt.sType, pt.required)
    }
  }

  def setRequired(newReq: Boolean): VirtualTypeWithReq = {
    val newR = r.copy(r.children).asInstanceOf[TypeWithRequiredness]
    newR.hardSetRequiredness(newReq)
    assert(newR.required == newReq)
    copy(r = newR)
  }

  override def toString: String = s"VirtualTypeWithReq($canonicalPType)"

  override def equals(obj: Any): Boolean = obj match {
    case t2: VirtualTypeWithReq => canonicalPType == t2.canonicalPType
    case _ => false
  }

  override def hashCode(): Int = {
    canonicalPType.hashCode() + 37
  }
}

sealed abstract class TypeWithRequiredness extends BaseTypeWithRequiredness {
  def _unionLiteral(a: Annotation): Unit
  def _unionPType(pType: PType): Unit
  def _unionEmitType(emitType: EmitType): Unit
  def _matchesPType(pt: PType): Boolean
  def unionLiteral(a: Annotation): Unit =
    if (a == null) union(false) else _unionLiteral(a)

  def fromPType(pType: PType): Unit = {
    union(pType.required)
    _unionPType(pType)
  }
  def fromEmitType(emitType: EmitType): Unit = {
    union(emitType.required)
    _unionEmitType(emitType)
  }
  def canonicalPType(t: Type): PType
  def canonicalEmitType(t: Type): EmitType = {
    t match {
      case TStream(element) => EmitType(SStream(this.asInstanceOf[RIterable].elementType.canonicalEmitType(element)), required)
      case _ =>
        val pt = canonicalPType(t)
        EmitType(pt.sType, pt.required)
    }
  }
  def matchesPType(pt: PType): Boolean = pt.required == required && _matchesPType(pt)
  def _toString: String
  override def toString: String = if (required) "+" + _toString else _toString
}

object RPrimitive {
  val children: IndexedSeq[TypeWithRequiredness] = FastSeq()
  val supportedTypes: Set[Type] = Set(TBoolean, TInt32, TInt64, TFloat32, TFloat64, TBinary, TString, TCall, TVoid, TRNGState)
  def typeSupported(t: Type): Boolean = RPrimitive.supportedTypes.contains(t) ||
    t.isInstanceOf[TLocus]
}

final case class RPrimitive() extends TypeWithRequiredness {
  val children: IndexedSeq[TypeWithRequiredness] = RPrimitive.children

  def _unionLiteral(a: Annotation): Unit = ()
  def _matchesPType(pt: PType): Boolean = RPrimitive.typeSupported(pt.virtualType)
  def _unionPType(pType: PType): Unit = assert(RPrimitive.typeSupported(pType.virtualType))
  def _unionEmitType(emitType: EmitType) = assert(RPrimitive.typeSupported(emitType.virtualType))
  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RPrimitive = {
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
  val children: IndexedSeq[TypeWithRequiredness] = FastSeq(elementType)
  def _unionLiteral(a: Annotation): Unit = {

    a.asInstanceOf[Iterable[_]].foreach(elt => elementType.unionLiteral(elt))
  }

  def _matchesPType(pt: PType): Boolean = elementType.matchesPType(tcoerce[PIterable](pt).elementType)
  def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PIterable].elementType)
  def _unionEmitType(emitType: EmitType): Unit = elementType.fromEmitType(emitType.st.asInstanceOf[SIndexablePointer].elementEmitType)
  def _toString: String = s"RIterable[${ elementType.toString }]"

  override def _maximizeChildren(): Unit = {
    if (eltRequired)
      elementType.children.foreach(_.maximize())
    else elementType.maximize()
  }

  override def _unionChildren(newChildren: IndexedSeq[BaseTypeWithRequiredness]): Unit = {
    val IndexedSeq(newEltReq) = newChildren
    unionElement(newEltReq)
  }

  override def _unionWithIntersection(ts: IndexedSeq[BaseTypeWithRequiredness]): Unit = {
    if (eltRequired) {
      var i = 0
      while(i < elementType.children.length) {
        elementType.children(i).unionWithIntersection(ts.map(t => tcoerce[RIterable](t).elementType.children(i)))
        i += 1
      }
    } else
      elementType.unionWithIntersection(ts.map(t => tcoerce[RIterable](t).elementType))
  }

  def unionElement(newElement: BaseTypeWithRequiredness): Unit = {
    if (eltRequired)
      (elementType.children, newElement.children).zipped.foreach { (r1, r2) => r1.unionFrom(r2) }
    else
      elementType.unionFrom(newElement)
  }

  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RIterable = {
    val IndexedSeq(newElt: TypeWithRequiredness) = newChildren
    RIterable(newElt)
  }
  def canonicalPType(t: Type): PType = {
    val elt = elementType.canonicalPType(tcoerce[TIterable](t).elementType)
    t match {
      case _: TArray => PCanonicalArray(elt, required = required)
      case _: TSet => PCanonicalSet(elt, required = required)
    }
  }
}
case class RDict(keyType: TypeWithRequiredness, valueType: TypeWithRequiredness)
  extends RIterable(RStruct.fromNamesAndTypes(Array("key" -> keyType, "value" -> valueType)), true) {
  override def _unionLiteral(a: Annotation): Unit =
    a.asInstanceOf[Map[_,_]].foreach { case (k, v) =>
      keyType.unionLiteral(k)
      valueType.unionLiteral(v)
    }
  override def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RDict = {
    val IndexedSeq(newElt: RStruct) = newChildren
    RDict(newElt.field("key"), newElt.field("value"))
  }
  override def canonicalPType(t: Type): PType =
    PCanonicalDict(
      keyType.canonicalPType(tcoerce[TDict](t).keyType),
      valueType.canonicalPType(tcoerce[TDict](t).valueType),
      required = required)
  override def _toString: String = s"RDict[${ keyType.toString }, ${ valueType.toString }]"
}
case class RNDArray(override val elementType: TypeWithRequiredness) extends RIterable(elementType, true) {
  override def _unionLiteral(a: Annotation): Unit = {
    val data = a.asInstanceOf[NDArray].getRowMajorElements()
    data.foreach { elt =>
      if (elt != null)
        elementType.unionLiteral(elt)
    }
  }
  override def _matchesPType(pt: PType): Boolean = elementType.matchesPType(tcoerce[PNDArray](pt).elementType)
  override def _unionPType(pType: PType): Unit = elementType.fromPType(pType.asInstanceOf[PNDArray].elementType)
  override def _unionEmitType(emitType: EmitType): Unit = elementType.fromEmitType(emitType.st.asInstanceOf[SNDArray].elementEmitType)
  override def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RNDArray = {
    val IndexedSeq(newElt: TypeWithRequiredness) = newChildren
    RNDArray(newElt)
  }
  override def canonicalPType(t: Type): PType = {
    val tnd = tcoerce[TNDArray](t)
    PCanonicalNDArray(elementType.canonicalPType(tnd.elementType), tnd.nDims, required = required)
  }
  override def _toString: String = s"RNDArray[${ elementType.toString }]"
}

case class RInterval(startType: TypeWithRequiredness, endType: TypeWithRequiredness) extends TypeWithRequiredness {
  val children: IndexedSeq[TypeWithRequiredness] = FastSeq(startType, endType)
  def _unionLiteral(a: Annotation): Unit = {
    startType.unionLiteral(a.asInstanceOf[Interval].start)
    endType.unionLiteral(a.asInstanceOf[Interval].end)
  }
  def _matchesPType(pt: PType): Boolean =
    startType.matchesPType(tcoerce[PInterval](pt).pointType) &&
      endType.matchesPType(tcoerce[PInterval](pt).pointType)
  def _unionPType(pType: PType): Unit = {
    startType.fromPType(pType.asInstanceOf[PInterval].pointType)
    endType.fromPType(pType.asInstanceOf[PInterval].pointType)
  }
  def _unionEmitType(emitType: EmitType): Unit = {
    val sInterval = emitType.st.asInstanceOf[SInterval]
    startType.fromEmitType(sInterval.pointEmitType)
    endType.fromEmitType(sInterval.pointEmitType)
  }
  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RInterval = {
    val IndexedSeq(newStart: TypeWithRequiredness, newEnd: TypeWithRequiredness) = newChildren
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
  def size: Int = fields.length
  val children: IndexedSeq[TypeWithRequiredness] = fields.map(_.typ)
  def _unionLiteral(a: Annotation): Unit =
    (children, a.asInstanceOf[Row].toSeq).zipped.foreach { (r, f) => r.unionLiteral(f) }
  def _matchesPType(pt: PType): Boolean =
    tcoerce[PBaseStruct](pt).fields.forall(f => children(f.index).matchesPType(f.typ))
  def _unionPType(pType: PType): Unit = {
    pType.asInstanceOf[PBaseStruct].fields.foreach(f => children(f.index).fromPType(f.typ))
  }
  def _unionEmitType(emitType: EmitType): Unit = {
    emitType.st.asInstanceOf[SBaseStruct].fieldEmitTypes.zipWithIndex.foreach{ case(et, idx) => children(idx).fromEmitType(et) }
  }

  def unionFields(other: RStruct): Unit = {
    assert(fields.length == other.fields.length)
    (fields, other.fields).zipped.foreach { (fd1, fd2) => fd1.typ.unionFrom(fd2.typ) }
  }

  def canonicalPType(t: Type): PType = t match {
    case ts: TStruct =>
      PCanonicalStruct(required = required,
        fields.map(f => f.name -> f.typ.canonicalPType(ts.fieldType(f.name))): _*)
    case ts: TTuple =>
      PCanonicalTuple((fields, ts._types).zipped.map { case(fr, ft) =>
        PTupleField(ft.index, fr.typ.canonicalPType(ft.typ))
      }, required = required)
  }
}

object RStruct {
  def fromNamesAndTypes(fields: IndexedSeq[(String, TypeWithRequiredness)]): RStruct =
    RStruct(fields.zipWithIndex.map { case ((name, typ), i) => RField(name, typ, i) })
}

case class RStruct(fields: IndexedSeq[RField]) extends RBaseStruct {
  val fieldType: collection.Map[String, TypeWithRequiredness] = toMapFast(fields)(_.name, _.typ)
  def field(name: String): TypeWithRequiredness = fieldType(name)
  def fieldOption(name: String): Option[TypeWithRequiredness] = fieldType.get(name)
  def hasField(name: String): Boolean = fieldType.contains(name)
  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RStruct = {
    assert(newChildren.length == fields.length)
    RStruct.fromNamesAndTypes(Array.tabulate(fields.length)(i => fields(i).name -> tcoerce[TypeWithRequiredness](newChildren(i))))
  }
  def select(newFields: Array[String]): RStruct =
    RStruct(Array.tabulate(newFields.length)(i => RField(newFields(i), field(newFields(i)), i)))
  def _toString: String = s"RStruct[${ fields.map(f => s"${ f.name }: ${ f.typ.toString }").mkString(",") }]"
}

object RTuple {
  def fromNamesAndTypes(fields: IndexedSeq[(String, TypeWithRequiredness)]): RTuple =
    RTuple(fields.zipWithIndex.map { case ((name, typ), i) => RField(name, typ, i) })
}

case class RTuple(fields: IndexedSeq[RField]) extends RBaseStruct {
  val fieldType: collection.Map[String, TypeWithRequiredness] = toMapFast(fields)(_.name, _.typ)
  def field(idx: Int): TypeWithRequiredness = fieldType(idx.toString)
  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RTuple = {
    assert(newChildren.length == fields.length)
    RTuple((fields, newChildren).zipped.map { (f, c) => RField(f.name, tcoerce[TypeWithRequiredness](c), f.index) })
  }
  def _toString: String = s"RTuple[${ fields.map(f => s"${ f.index }: ${ f.typ.toString }").mkString(",") }]"
}

case class RUnion(cases: IndexedSeq[(String, TypeWithRequiredness)]) extends TypeWithRequiredness {
  val children: IndexedSeq[TypeWithRequiredness] = cases.map(_._2)
  def _unionLiteral(a: Annotation): Unit = ???
  def _matchesPType(pt: PType): Boolean = ???
  def _unionPType(pType: PType): Unit = ???
  def _unionEmitType(emitType: EmitType): Unit = ???
  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RUnion = {
    assert(newChildren.length == cases.length)
    RUnion(Array.tabulate(cases.length)(i => cases(i)._1 -> tcoerce[TypeWithRequiredness](newChildren(i))))
  }
  def canonicalPType(t: Type): PType = ???
  def _toString: String = s"RStruct[${ cases.map { case (n, t) => s"${ n }: ${ t.toString }" }.mkString(",") }]"
}

object RTable {
  def apply(rowStruct: RStruct, globStruct: RStruct, key: IndexedSeq[String]): RTable = {
    RTable(rowStruct.fields.map(f => f.name -> f.typ), globStruct.fields.map(f => f.name -> f.typ), key)
  }

  def fromTableStage(ec: ExecuteContext, s: TableStage): RTable = {
    def virtualTypeWithReq(ir: IR, inputs: Env[PType]): VirtualTypeWithReq = {
      import is.hail.expr.ir.Requiredness
      val ns = ir.noSharing
      val usesAndDefs = ComputeUsesAndDefs(ns, errorIfFreeVariables = false)
      val req = Requiredness.apply(ns, usesAndDefs, ec, inputs)
      VirtualTypeWithReq(ir.typ, req.lookup(ns).asInstanceOf[TypeWithRequiredness])
    }

    // requiredness uses ptypes for legacy reasons, there is a 1-1 mapping between
    // RTypes and canonical PTypes
    val letBindingReq =
      s.letBindings.foldLeft(Env.empty[PType]) { case (env, (name, ir)) =>
        env.bind(name, virtualTypeWithReq(ir, env).canonicalPType)
      }

    val broadcastValBindings =
      Env.fromSeq(s.broadcastVals.map { case (name, ir) =>
        (name, virtualTypeWithReq(ir, letBindingReq).canonicalPType)
      })

    val ctxReq =
      VirtualTypeWithReq(TIterable.elementType(s.contexts.typ),
        virtualTypeWithReq(s.contexts, letBindingReq).r.asInstanceOf[RIterable].elementType
      )

    val globalRType =
      virtualTypeWithReq(s.globals, letBindingReq).r.asInstanceOf[RStruct]

    val rowRType =
      virtualTypeWithReq(s.partitionIR, broadcastValBindings.bind(s.ctxRefName, ctxReq.canonicalPType))
        .r.asInstanceOf[RIterable].elementType.asInstanceOf[RStruct]

    RTable(rowRType, globalRType, s.kType.fieldNames)
  }
}
case class RTable(rowFields: IndexedSeq[(String, TypeWithRequiredness)], globalFields: IndexedSeq[(String, TypeWithRequiredness)], key: Seq[String]) extends BaseTypeWithRequiredness {
  val rowTypes: IndexedSeq[TypeWithRequiredness] = rowFields.map(_._2)
  val globalTypes: IndexedSeq[TypeWithRequiredness] = globalFields.map(_._2)

  val keyFields: Set[String] = key.toSet
  val valueFields: Set[String] = rowFields.map(_._1).filter(n => !keyFields.contains(n)).toSet

  val fieldMap: Map[String, TypeWithRequiredness] = (rowFields ++ globalFields).toMap
  def field(name: String): TypeWithRequiredness = fieldMap(name)

  val children: IndexedSeq[TypeWithRequiredness] = rowTypes ++ globalTypes

  val rowType: RStruct = RStruct.fromNamesAndTypes(rowFields)
  val globalType: RStruct = RStruct.fromNamesAndTypes(globalFields)

  def unionRows(req: RStruct): Unit = rowFields.foreach { case (n, r) => if (req.hasField(n)) r.unionFrom(req.field(n)) }
  def unionRows(req: RTable): Unit = unionRows(req.rowType)

  def unionGlobals(req: RStruct): Unit = globalFields.foreach { case (n, r) => if (req.hasField(n)) r.unionFrom(req.field(n)) }
  def unionGlobals(req: RTable): Unit = unionGlobals(req.globalType)

  def unionKeys(req: RStruct): Unit = key.foreach { n => field(n).unionFrom(req.field(n)) }
  def unionKeys(req: RTable): Unit = {
    assert(key.length <= req.key.length)
    (key, req.key).zipped.foreach { (k, rk) => field(k).unionFrom(req.field(rk)) }
  }

  def unionValues(req: RStruct): Unit = valueFields.foreach { n => if (req.hasField(n)) field(n).unionFrom(req.field(n)) }
  def unionValues(req: RTable): Unit = unionValues(req.rowType)

  def changeKey(key: IndexedSeq[String]): RTable = RTable(rowFields, globalFields, key)

  def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): RTable = {
    assert(newChildren.length == rowFields.length + globalFields.length)
    val newRowFields = (rowFields, newChildren.take(rowFields.length)).zipped.map { case ((n, _), r: TypeWithRequiredness) => n -> r }
    val newGlobalFields = (globalFields, newChildren.drop(rowFields.length)).zipped.map { case ((n, _), r: TypeWithRequiredness) => n -> r }
    RTable(newRowFields, newGlobalFields, key)
  }

  def asMatrixType(colField: String, entryField: String): RMatrix = {
    val row = RStruct.fromNamesAndTypes(rowFields.filter(_._1 != entryField))
    val entry = tcoerce[RStruct](tcoerce[RIterable](field(entryField)).elementType)
    val col = tcoerce[RStruct](tcoerce[RIterable](field(colField)).elementType)
    val global = RStruct.fromNamesAndTypes(globalFields.filter(_._1 != colField))
    RMatrix(row, entry, col, global)
  }

  override def toString: String =
    s"RTable[\n  row:${ rowType.toString }\n  global:${ globalType.toString }]"
}

case class RMatrix(rowType: RStruct, entryType: RStruct, colType: RStruct, globalType: RStruct) {
  val entriesRVType: RStruct = RStruct.fromNamesAndTypes(FastSeq(MatrixType.entriesIdentifier -> RIterable(entryType)))
}

case class RBlockMatrix(elementType: TypeWithRequiredness) extends BaseTypeWithRequiredness {
  override def children: IndexedSeq[BaseTypeWithRequiredness] = FastSeq(elementType)

  override def copy(newChildren: IndexedSeq[BaseTypeWithRequiredness]): BaseTypeWithRequiredness = RBlockMatrix(newChildren(0).asInstanceOf[TypeWithRequiredness])

  override def toString: String = s"RBlockMatrix(${elementType})"
}
