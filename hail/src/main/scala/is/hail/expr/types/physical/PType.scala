package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.types.virtual._
import is.hail.expr.types.{BaseType, Requiredness}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

class PTypeSerializer extends CustomSerializer[PType](format => (
  { case JString(s) => PType.canonical(IRParser.parsePType(s)) },
  { case t: PType => JString(t.toString) }))


object PType {
  def genScalar(required: Boolean): Gen[PType] =
    Gen.oneOf(PBoolean(required), PInt32(required), PInt64(required), PFloat32(required),
      PFloat64(required), PString(required), PCall(required))

  val genOptionalScalar: Gen[PType] = genScalar(false)

  val genRequiredScalar: Gen[PType] = genScalar(true)

  def genComplexType(required: Boolean): Gen[ComplexPType] = {
    val rgDependents = ReferenceGenome.references.values.toArray.map(rg =>
      PLocus(rg, required))
    val others = Array(PCall(required))
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
      PStruct(fields, required)
  }

  def preGenTuple(required: Boolean, genFieldType: Gen[PType]): Gen[PTuple] = {
    for (fields <- genFields(required, genFieldType)) yield
      PTuple(required, fields.map(_.typ): _*)
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
      Gen.const(PStruct.empty(required))
    else if (size < 2)
      genScalar(required)
    else {
      Gen.frequency(
        (4, genScalar(required)),
        (1, genComplexType(required)),
        (1, genArb.map {
          PArray(_)
        }),
        (1, genArb.map {
          PSet(_)
        }),
        (1, genArb.map {
          PInterval(_)
        }),
        (1, preGenTuple(required, genArb)),
        (1, Gen.zip(genRequired, genArb).map { case (k, v) => PDict(k, v) }),
        (1, genPStruct.resize(size)))
    }

  def preGenArb(required: Boolean, genStruct: Gen[PStruct] = genStruct): Gen[PType] =
    Gen.sized(genSized(_, required, genStruct))

  def genArb: Gen[PType] = Gen.coin(0.2).flatMap(preGenArb(_))

  val genOptional: Gen[PType] = preGenArb(required = false)

  val genRequired: Gen[PType] = preGenArb(required = true)

  val genInsertable: Gen[PStruct] = genInsertableStruct

  implicit def arbType = Arbitrary(genArb)

  def canonical(t: Type, required: Boolean): PType = {
    t match {
      case TInt32 => PInt32(required)
      case TInt64 => PInt64(required)
      case TFloat32 => PFloat32(required)
      case TFloat64 => PFloat64(required)
      case TBoolean => PBoolean(required)
      case TBinary => PBinary(required)
      case TString => PString(required)
      case TCall => PCall(required)
      case t: TLocus => PLocus(t.rg, required)
      case t: TInterval => PInterval(canonical(t.pointType), required)
      case t: TStream => PStream(canonical(t.elementType), required)
      case t: TArray => PArray(canonical(t.elementType), required)
      case t: TSet => PSet(canonical(t.elementType), required)
      case t: TDict => PDict(canonical(t.keyType), canonical(t.valueType), required)
      case t: TTuple => PTuple(t._types.map(tf => PTupleField(tf.index, canonical(tf.typ))), required)
      case t: TStruct => PStruct(t.fields.map(f => PField(f.name, canonical(f.typ), f.index)), required)
      case t: TNDArray => PNDArray(canonical(t.elementType).setRequired(true), t.nDims, required)
      case TVoid => PVoid
    }
  }

  def canonical(t: Type): PType = canonical(t, false)

  // currently identity
  def canonical(t: PType): PType = {
    t match {
      case t: PInt32 => PInt32(t.required)
      case t: PInt64 => PInt64(t.required)
      case t: PFloat32 => PFloat32(t.required)
      case t: PFloat64 => PFloat64(t.required)
      case t: PBoolean => PBoolean(t.required)
      case t: PBinary => PBinary(t.required)
      case t: PString => PString(t.required)
      case t: PCall => PCall(t.required)
      case t: PLocus => PLocus(t.rg, t.required)
      case t: PInterval => PInterval(canonical(t.pointType), t.required)
      case t: PStream => PStream(canonical(t.elementType), t.required)
      case t: PArray => PArray(canonical(t.elementType), t.required)
      case t: PSet => PSet(canonical(t.elementType), t.required)
      case t: PTuple => PTuple(t._types.map(pf => PTupleField(pf.index, canonical(pf.typ))), t.required)
      case t: PStruct => PStruct(t.fields.map(f => PField(f.name, canonical(f.typ), f.index)), t.required)
      case t: PNDArray => PNDArray(canonical(t.elementType), t.nDims, t.required)
      case t: PDict => PDict(canonical(t.keyType), canonical(t.valueType), t.required)
      case PVoid => PVoid
    }
  }
}

abstract class PType extends Serializable with Requiredness {
  self =>

  def genValue: Gen[Annotation] =
    if (required) genNonmissingValue else Gen.nextCoin(0.05).flatMap(isEmpty => if (isEmpty) Gen.const(null) else genNonmissingValue)

  def genNonmissingValue: Gen[Annotation] = virtualType.genNonmissingValue

  def virtualType: Type

  override def toString: String = {
    val sb = new StringBuilder
    pretty(sb, 0, true)
    sb.result()
  }

  def unsafeOrdering(): UnsafeOrdering = ???

  def isCanonical: Boolean = PType.canonical(this) == this // will recons, may need to rewrite this method

  def unsafeOrdering(rightType: PType): UnsafeOrdering = {
    require(virtualType == rightType.virtualType, s"$this, $rightType")
    unsafeOrdering()
  }

  def asIdent: String = (if (required) "r_" else "o_") + _asIdent

  def _asIdent: String

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (required)
      sb.append("+")
    _pretty(sb, indent, compact)
  }

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean)

  def codeOrdering(mb: EmitMethodBuilder[_]): CodeOrdering =
    codeOrdering(mb, this)

  def codeOrdering(mb: EmitMethodBuilder[_], so: SortOrder): CodeOrdering =
    codeOrdering(mb, this, so)

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: SortOrder): CodeOrdering =
    so match {
      case Ascending => codeOrdering(mb, other)
      case Descending => codeOrdering(mb, other).reverse
    }

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering

  def byteSize: Long = 1

  def alignment: Long = byteSize

  /*  Fundamental types are types that can be handled natively by RegionValueBuilder: primitive
      types, Array and Struct. */
  def fundamentalType: PType = this

  final def unary_+(): PType = setRequired(true)

  final def unary_-(): PType = setRequired(false)

  def setRequired(required: Boolean): PType

  final def orMissing(required2: Boolean): PType = {
    if (!required2)
      setRequired(false)
    else
      this
  }

  final def isOfType(t: PType): Boolean = this.virtualType == t.virtualType

  final def isPrimitive: Boolean =
    fundamentalType.isInstanceOf[PBoolean] || isNumeric

  final def isNumeric: Boolean =
    fundamentalType.isInstanceOf[PInt32] ||
      fundamentalType.isInstanceOf[PInt64] ||
      fundamentalType.isInstanceOf[PFloat32] ||
      fundamentalType.isInstanceOf[PFloat64]

  def containsPointers: Boolean = false

  def subsetTo(t: Type): PType = {
    this match {
      case PCanonicalStruct(fields, r) =>
        val ts = t.asInstanceOf[TStruct]
        PCanonicalStruct(r, fields.flatMap { pf => ts.fieldOption(pf.name).map { vf => (pf.name, pf.typ.subsetTo(vf.typ)) } }: _*)
      case PCanonicalTuple(fields, r) =>
        val tt = t.asInstanceOf[TTuple]
        PCanonicalTuple(fields.flatMap { pf => tt.fieldIndex.get(pf.index).map(vi => PTupleField(vi, pf.typ.subsetTo(tt.types(vi)))) }, r)
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

  def deepInnerRequired(required: Boolean): PType =
    this match {
      case t: PArray => PArray(t.elementType.deepInnerRequired(true), required)
      case t: PSet => PSet(t.elementType.deepInnerRequired(true), required)
      case t: PDict => PDict(t.keyType.deepInnerRequired(true), t.valueType.deepInnerRequired(true), required)
      case t: PStruct =>
        PStruct(t.fields.map(f => PField(f.name, f.typ.deepInnerRequired(true), f.index)), required)
      case t: PCanonicalTuple =>
        PCanonicalTuple(t._types.map { f => f.copy(typ = f.typ.deepInnerRequired(true)) }, required)
      case t: PInterval =>
        PInterval(t.pointType.deepInnerRequired(true), required)
      case t =>
        t.setRequired(required)
    }

  // Semantics: must be callable without requiredeness check: srcAddress must point to non-null value
  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long]

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_]

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_]): Code[_] =
    this.copyFromTypeAndStackValue(mb, region, srcPType, stackValue, false)

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit]
  def constructAtAddressFromValue(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, src: Code[_], deepCopy: Boolean): Code[Unit]
    = constructAtAddress(mb, addr, region, srcPType, coerce[Long](src), deepCopy)

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit

  def deepRename(t: Type) = this

  def defaultValue: PCode = PCode(this, ir.defaultValue(this))

  def copyFromPValue(mb: EmitMethodBuilder[_], region: Value[Region], pv: PCode): PCode =
    PCode(this, copyFromTypeAndStackValue(mb, region, pv.pt, pv.code))

  final def typeCheck(a: Any): Boolean = a == null || _typeCheck(a)

  def _typeCheck(a: Any): Boolean = virtualType._typeCheck(a)

  def load(src: Code[Long]): PCode = PCode(this, Region.loadIRIntermediate(this)(src))
}
