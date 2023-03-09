package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailStateManager}
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.ir._
import is.hail.types.physical.stypes.concrete.SRNGState
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.virtual._
import is.hail.types.{Requiredness, tcoerce}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

class PTypeSerializer extends CustomSerializer[PType](format => (
  { case JString(s) => PType.canonical(IRParser.parsePType(s)) },
  { case t: PType => JString(t.toString) }))

class PStructSerializer extends CustomSerializer[PStruct](format => (
  { case JString(s) => tcoerce[PStruct](IRParser.parsePType(s)) },
  { case t: PStruct => JString(t.toString) }))

object PType {
  def genScalar(required: Boolean): Gen[PType] =
    Gen.oneOf(PBoolean(required), PInt32(required), PInt64(required), PFloat32(required),
      PFloat64(required), PCanonicalString(required), PCanonicalCall(required))

  val genOptionalScalar: Gen[PType] = genScalar(false)

  val genRequiredScalar: Gen[PType] = genScalar(true)

  def genComplexType(required: Boolean): Gen[PType] = {
    val rgDependents = ReferenceGenome.hailReferences.toArray.map(PCanonicalLocus(_, required))
    val others = Array(PCanonicalCall(required))
    Gen.oneOfSeq(rgDependents ++ others)
  }

  def genFields(required: Boolean, genFieldType: Gen[PType]): Gen[Array[PField]] = {
    Gen.buildableOf[Array](
      Gen.zip(Gen.identifier, genFieldType))
      .filter(fields => fields.map(_._1).areDistinct())
      .map(fields => fields
        .iterator
        .zipWithIndex
        .map { case ((k, t), i) => PField(k, t, i) }
        .toArray)
  }

  def preGenStruct(required: Boolean, genFieldType: Gen[PType]): Gen[PStruct] = {
    for (fields <- genFields(required, genFieldType)) yield
      PCanonicalStruct(fields, required)
  }

  def preGenTuple(required: Boolean, genFieldType: Gen[PType]): Gen[PTuple] = {
    for (fields <- genFields(required, genFieldType)) yield
      PCanonicalTuple(required, fields.map(_.typ): _*)
  }

  private val defaultRequiredGenRatio = 0.2

  def genStruct: Gen[PStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(preGenStruct(_, genArb))

  val genOptionalStruct: Gen[PType] = preGenStruct(required = false, genArb)

  val genRequiredStruct: Gen[PType] = preGenStruct(required = true, genArb)

  val genInsertableStruct: Gen[PStruct] = Gen.coin(defaultRequiredGenRatio).flatMap(required =>
    if (required)
      preGenStruct(required = true, genArb)
    else
      preGenStruct(required = false, genOptional))

  def genSized(size: Int, required: Boolean, genPStruct: Gen[PStruct]): Gen[PType] =
    if (size < 1)
      Gen.const(PCanonicalStruct.empty(required))
    else if (size < 2)
      genScalar(required)
    else {
      Gen.frequency(
        (4, genScalar(required)),
        (1, genComplexType(required)),
        (1, genArb.map {
          PCanonicalArray(_)
        }),
        (1, genArb.map {
          PCanonicalSet(_)
        }),
        (1, genArb.map {
          PCanonicalInterval(_)
        }),
        (1, preGenTuple(required, genArb)),
        (1, Gen.zip(genRequired, genArb).map { case (k, v) => PCanonicalDict(k, v) }),
        (1, genPStruct.resize(size)))
    }

  def preGenArb(required: Boolean, genStruct: Gen[PStruct] = genStruct): Gen[PType] =
    Gen.sized(genSized(_, required, genStruct))

  def genArb: Gen[PType] = Gen.coin(0.2).flatMap(preGenArb(_))

  val genOptional: Gen[PType] = preGenArb(required = false)

  val genRequired: Gen[PType] = preGenArb(required = true)

  val genInsertable: Gen[PStruct] = genInsertableStruct

  implicit def arbType = Arbitrary(genArb)

  def canonical(t: Type, required: Boolean, innerRequired: Boolean): PType = {
    t match {
      case TInt32 => PInt32(required)
      case TInt64 => PInt64(required)
      case TFloat32 => PFloat32(required)
      case TFloat64 => PFloat64(required)
      case TBoolean => PBoolean(required)
      case TBinary => PCanonicalBinary(required)
      case TString => PCanonicalString(required)
      case TCall => PCanonicalCall(required)
      case TRNGState => StoredSTypePType(SRNGState(None), required)
      case t: TLocus => PCanonicalLocus(t.rg, required)
      case t: TInterval => PCanonicalInterval(canonical(t.pointType, innerRequired, innerRequired), required)
      case t: TArray => PCanonicalArray(canonical(t.elementType, innerRequired, innerRequired), required)
      case t: TSet => PCanonicalSet(canonical(t.elementType, innerRequired, innerRequired), required)
      case t: TDict => PCanonicalDict(canonical(t.keyType, innerRequired, innerRequired), canonical(t.valueType, innerRequired, innerRequired), required)
      case t: TTuple => PCanonicalTuple(t._types.map(tf => PTupleField(tf.index, canonical(tf.typ, innerRequired, innerRequired))), required)
      case t: TStruct => PCanonicalStruct(t.fields.map(f => PField(f.name, canonical(f.typ, innerRequired, innerRequired), f.index)), required)
      case t: TNDArray => PCanonicalNDArray(canonical(t.elementType, innerRequired, innerRequired).setRequired(true), t.nDims, required)
      case TVoid => PVoid
    }
  }

  def canonical(t: Type, required: Boolean): PType = canonical(t, required, false)

  def canonical(t: Type): PType = canonical(t, false, false)

  // currently identity
  def canonical(t: PType): PType = {
    t match {
      case t: PInt32 => PInt32(t.required)
      case t: PInt64 => PInt64(t.required)
      case t: PFloat32 => PFloat32(t.required)
      case t: PFloat64 => PFloat64(t.required)
      case t: PBoolean => PBoolean(t.required)
      case t: PBinary => PCanonicalBinary(t.required)
      case t: PString => PCanonicalString(t.required)
      case t: PCall => PCanonicalCall(t.required)
      case t: PLocus => PCanonicalLocus(t.rg, t.required)
      case t: PInterval => PCanonicalInterval(canonical(t.pointType), t.required)
      case t: PArray => PCanonicalArray(canonical(t.elementType), t.required)
      case t: PSet => PCanonicalSet(canonical(t.elementType), t.required)
      case t: PTuple => PCanonicalTuple(t._types.map(pf => PTupleField(pf.index, canonical(pf.typ))), t.required)
      case t: PStruct => PCanonicalStruct(t.fields.map(f => PField(f.name, canonical(f.typ), f.index)), t.required)
      case t: PNDArray => PCanonicalNDArray(canonical(t.elementType), t.nDims, t.required)
      case t: PDict => PCanonicalDict(canonical(t.keyType), canonical(t.valueType), t.required)
      case PVoid => PVoid
    }
  }

  def literalPType(t: Type, a: Annotation): PType = {
    val rb = new BooleanArrayBuilder()
    val crib = new IntArrayBuilder()
    val cib = new IntArrayBuilder()

    def indexTypes(t: Type): Unit = {
      val ci = crib.size

      rb += true

      t match {
        case t: TSet =>
          indexTypes(t.elementType)
        case t: TDict =>
          crib += 0
          cib += 0
          indexTypes(t.keyType)
          crib(ci) = rb.size
          cib(ci) = crib.size
          indexTypes(t.valueType)
        case t: TArray =>
          indexTypes(t.elementType)
        case t: TStream =>
          indexTypes(t.elementType)
        case t: TInterval =>
          indexTypes(t.pointType)
        case t: TNDArray =>
          indexTypes(t.elementType)
        case t: TBaseStruct =>
          val n = t.size

          crib.setSizeUninitialized(ci + n)
          cib.setSizeUninitialized(ci + n)
          cib.setSize(ci + n)

          var j = 0
          while (j < n) {
            crib(ci + j) = rb.size
            cib(ci + j) = crib.size
            indexTypes(t.types(j))
            j += 1
          }
        case _ =>
      }
    }

    indexTypes(t)

    val requiredVector = rb.result()
    val childRequiredIndex = crib.result()
    val childIndex = cib.result()

    def setOptional(t: Type, a: Annotation, ri: Int, ci: Int): Unit = {
      if (a == null) {
        requiredVector(ri) = false
        return
      }

      t match {
        case t: TSet =>
          a.asInstanceOf[Set[_]].iterator
            .foreach(x => setOptional(t.elementType, x, ri + 1, ci))
        case t: TDict =>
          a.asInstanceOf[Map[_, _]].iterator
            .foreach { case (k, v) =>
              setOptional(t.keyType, k, ri + 1, ci + 1)
              setOptional(t.valueType, v, childRequiredIndex(ci), childIndex(ci))
            }
        case t: TArray =>
          a.asInstanceOf[IndexedSeq[_]].iterator
            .foreach(x => setOptional(t.elementType, x, ri + 1, ci))
        case t: TStream =>
          a.asInstanceOf[IndexedSeq[_]].iterator
            .foreach(x => setOptional(t.elementType, x, ri + 1, ci))
        case t: TInterval =>
          val i = a.asInstanceOf[Interval]
          setOptional(t.pointType, i.start, ri + 1, ci)
          setOptional(t.pointType, i.end, ri + 1, ci)
        case t: TNDArray =>
          val r = a.asInstanceOf[Row]
          val elems = r(2).asInstanceOf[IndexedSeq[_]]
          elems.foreach { x =>
            setOptional(t.elementType, x, ri + 1, ci)
          }
        case t: TBaseStruct =>
          val r = a.asInstanceOf[Row]
          val n = r.size

          var j = 0
          while (j < n) {
            setOptional(t.types(j), r(j), childRequiredIndex(ci + j), childIndex(ci + j))
            j += 1
          }
        case _ =>
      }
    }

    setOptional(t, a, 0, 0)

    def canonical(t: Type, ri: Int, ci: Int): PType = {
      t match {
        case TBinary => PCanonicalBinary(requiredVector(ri))
        case TBoolean => PBoolean(requiredVector(ri))
        case TVoid => PVoid
        case t: TSet =>
          PCanonicalSet(canonical(t.elementType, ri + 1, ci), requiredVector(ri))
        case t: TDict =>
          PCanonicalDict(
            canonical(t.keyType, ri + 1, ci + 1),
            canonical(t.valueType, childRequiredIndex(ci), childIndex(ci)),
            requiredVector(ri))
        case t: TArray =>
          PCanonicalArray(canonical(t.elementType, ri + 1, ci), requiredVector(ri))
        case t: TStream =>
          PCanonicalArray(canonical(t.elementType, ri + 1, ci), requiredVector(ri))
        case TInt32 => PInt32(requiredVector(ri))
        case TInt64 => PInt64(requiredVector(ri))
        case TFloat32 => PFloat32(requiredVector(ri))
        case TFloat64 => PFloat64(requiredVector(ri))
        case t: TInterval =>
          PCanonicalInterval(canonical(t.pointType, ri + 1, ci), requiredVector(ri))
        case t: TLocus => PCanonicalLocus(t.rg, requiredVector(ri))
        case TCall => PCanonicalCall(requiredVector(ri))
        case t: TNDArray =>
          PCanonicalNDArray(canonical(t.elementType, ri + 1, ci), t.nDims, requiredVector(ri))
        case TString => PCanonicalString(requiredVector(ri))
        case t: TStruct =>
          PCanonicalStruct(requiredVector(ri),
            t.fields.zipWithIndex.map { case (f, j) =>
              f.name -> canonical(f.typ, childRequiredIndex(ci + j), childIndex(ci + j))
            }: _*)
        case t: TTuple =>
          PCanonicalTuple(requiredVector(ri),
            t.types.zipWithIndex.map { case (ft, j) =>
              canonical(ft, childRequiredIndex(ci + j), childIndex(ci + j))
            }: _*)
      }
    }

    canonical(t, 0, 0)
  }

  def canonicalize(t: PType, ctx: ExecuteContext, path: List[String]): Option[(HailClassLoader) => AsmFunction2RegionLongLong] = {
    def canonicalPath(pt: PType, path: List[String]): PType = {
      if (path.isEmpty) {
        PType.canonical(pt)
      }

      val head :: tail = path
      pt match {
        case t@PCanonicalStruct(fields, required) =>
          assert(t.hasField(head))
          PCanonicalStruct(fields.map(f => if (f.name == head) f.copy(typ = canonicalPath(f.typ, tail)) else f), required)
        case PCanonicalArray(element, required) =>
          assert(head == "element")
          PCanonicalArray(canonicalPath(element, tail), required)
        case other =>
          throw new RuntimeException(s"cannot canonicalize nested path under type $other")
      }
    }

    val cpt = canonicalPath(t, path)
    if (cpt == t)
      None
    else {
      val fb = EmitFunctionBuilder[AsmFunction2RegionLongLong](ctx,
        "copyFromAddr",
        FastIndexedSeq[ParamType](classInfo[Region], LongInfo), LongInfo)

      fb.emitWithBuilder { cb =>
        val region = fb.apply_method.getCodeParam[Region](1)
        val srcAddr = fb.apply_method.getCodeParam[Long](2)
        cpt.store(cb, region, t.loadCheapSCode(cb, srcAddr), deepCopy = false)
      }
      Some(fb.result(ctx))
    }
  }
}

abstract class PType extends Serializable with Requiredness {
  self =>

  def genValue(sm: HailStateManager): Gen[Annotation] =
    if (required) genNonmissingValue(sm) else Gen.nextCoin(0.05).flatMap(isEmpty => if (isEmpty) Gen.const(null) else genNonmissingValue(sm))

  def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = virtualType.genNonmissingValue(sm)

  def virtualType: Type

  def sType: SType

  def copiedType: PType

  override def toString: String = {
    val sb = new StringBuilder
    pretty(sb, 0, true)
    sb.result()
  }

  def unsafeOrdering(sm: HailStateManager): UnsafeOrdering

  def isCanonical: Boolean = PType.canonical(this) == this // will recons, may need to rewrite this method

  def unsafeOrdering(sm: HailStateManager, rightType: PType): UnsafeOrdering = {
    require(virtualType == rightType.virtualType, s"$this, $rightType")
    unsafeOrdering(sm)
  }

  def asIdent: String = (if (required) "r_" else "o_") + _asIdent

  def _asIdent: String

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (required)
      sb.append("+")
    _pretty(sb, indent, compact)
  }

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean)

  def byteSize: Long

  def alignment: Long = byteSize

  final def unary_+(): PType = setRequired(true)

  final def unary_-(): PType = setRequired(false)

  def setRequired(required: Boolean): PType

  def equalModuloRequired(that: PType): Boolean = this == that.setRequired(required)

  final def orMissing(required2: Boolean): PType = {
    if (!required2)
      setRequired(false)
    else
      this
  }

  final def isOfType(t: PType): Boolean = this.virtualType == t.virtualType

  final def isPrimitive: Boolean =
    isInstanceOf[PBoolean] || isNumeric

  final def isRealizable: Boolean = !isInstanceOf[PUnrealizable]

  final def isNumeric: Boolean =
    isInstanceOf[PInt32] ||
      isInstanceOf[PInt64] ||
      isInstanceOf[PFloat32] ||
      isInstanceOf[PFloat64]

  def containsPointers: Boolean

  def subsetTo(t: Type): PType = {
    this match {
      case x@PCanonicalStruct(fields, r) =>
        val ts = t.asInstanceOf[TStruct]
        assert(ts.fieldNames.forall(x.fieldNames.contains))
        PCanonicalStruct(r, fields.flatMap { pf => ts.fieldOption(pf.name).map { vf => (pf.name, pf.typ.subsetTo(vf.typ)) } }: _*)
      case PCanonicalTuple(fields, r) =>
        val tt = t.asInstanceOf[TTuple]
        PCanonicalTuple(fields.flatMap { pf => tt.fieldIndex.get(pf.index).map(vi => PTupleField(pf.index, pf.typ.subsetTo(tt.types(vi)))) }, r)
      case PCanonicalArray(e, r) =>
        val ta = t.asInstanceOf[TArray]
        PCanonicalArray(e.subsetTo(ta.elementType), r)
      case PCanonicalSet(e, r) =>
        val ts = t.asInstanceOf[TSet]
        PCanonicalSet(e.subsetTo(ts.elementType), r)
      case PCanonicalDict(k, v, r) =>
        val td = t.asInstanceOf[TDict]
        PCanonicalDict(k.subsetTo(td.keyType), v.subsetTo(td.valueType), r)
      case PCanonicalInterval(p, r) =>
        val ti = t.asInstanceOf[TInterval]
        PCanonicalInterval(p.subsetTo(ti.pointType), r)
      case _ =>
        assert(virtualType == t)
        this
    }
  }

  protected[physical] def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long

  def copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    // no requirement for requiredness
    // this can have more/less requiredness than srcPType
    // if value is not compatible with this, an exception will be thrown
    (virtualType, srcPType.virtualType) match {
      case (l: TBaseStruct, r: TBaseStruct) => assert(l.isCompatibleWith(r))
      case _ => assert(virtualType == srcPType.virtualType, s"virtualType: ${virtualType} != srcPType.virtualType: ${srcPType.virtualType}")
    }
    _copyFromAddress(sm, region, srcPType, srcAddress, deepCopy)
  }

  // return a SCode that can cheaply operate on the region representation. Generally a pointer type, but not necessarily (e.g. primitives).
  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SValue

  // stores a stack value as a region value of this type
  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long]

  // stores a stack value inside pre-allocated memory of this type (in a nested structure, for instance).
  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit

  def unstagedStoreAtAddress(sm: HailStateManager, addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit

  def deepRename(t: Type): PType = this

  // called to load a region value's start address from a nested representation.
  // Usually a no-op, but may need to dereference a pointer.
  def loadFromNested(addr: Code[Long]): Code[Long]

  def unstagedLoadFromNested(addr: Long): Long

  def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long

  def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit
}
