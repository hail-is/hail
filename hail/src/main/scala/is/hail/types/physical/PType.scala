package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s
import is.hail.asm4s._
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.types.{coerce, Requiredness}
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

class PTypeSerializer extends CustomSerializer[PType](format => (
  { case JString(s) => PType.canonical(IRParser.parsePType(s)) },
  { case t: PType => JString(t.toString) }))

class PStructSerializer extends CustomSerializer[PStruct](format => (
  { case JString(s) => coerce[PStruct](IRParser.parsePType(s)) },
  { case t: PStruct => JString(t.toString) }))

object PType {
  def genScalar(required: Boolean): Gen[PType] =
    Gen.oneOf(PBoolean(required), PInt32(required), PInt64(required), PFloat32(required),
      PFloat64(required), PCanonicalString(required), PCanonicalCall(required))

  val genOptionalScalar: Gen[PType] = genScalar(false)

  val genRequiredScalar: Gen[PType] = genScalar(true)

  def genComplexType(required: Boolean): Gen[ComplexPType] = {
    val rgDependents = ReferenceGenome.references.values.toArray.map(rg =>
      PCanonicalLocus(rg, required))
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

  def canonical(t: Type, required: Boolean): PType = {
    t match {
      case TInt32 => PInt32(required)
      case TInt64 => PInt64(required)
      case TFloat32 => PFloat32(required)
      case TFloat64 => PFloat64(required)
      case TBoolean => PBoolean(required)
      case TBinary => PCanonicalBinary(required)
      case t: TShuffle => PCanonicalShuffle(t, required)
      case TString => PCanonicalString(required)
      case TCall => PCanonicalCall(required)
      case t: TLocus => PCanonicalLocus(t.rg, required)
      case t: TInterval => PCanonicalInterval(canonical(t.pointType), required)
      case t: TStream => PCanonicalStream(canonical(t.elementType), required)
      case t: TArray => PCanonicalArray(canonical(t.elementType), required)
      case t: TSet => PCanonicalSet(canonical(t.elementType), required)
      case t: TDict => PCanonicalDict(canonical(t.keyType), canonical(t.valueType), required)
      case t: TTuple => PCanonicalTuple(t._types.map(tf => PTupleField(tf.index, canonical(tf.typ))), required)
      case t: TStruct => PCanonicalStruct(t.fields.map(f => PField(f.name, canonical(f.typ), f.index)), required)
      case t: TNDArray => PCanonicalNDArray(canonical(t.elementType).setRequired(true), t.nDims, required)
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
      case t: PBinary => PCanonicalBinary(t.required)
      case t: PString => PCanonicalString(t.required)
      case t: PCall => PCanonicalCall(t.required)
      case t: PLocus => PCanonicalLocus(t.rg, t.required)
      case t: PInterval => PCanonicalInterval(canonical(t.pointType), t.required)
      case t: PStream => PCanonicalStream(canonical(t.elementType), t.required)
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
    val rb = new ArrayBuilder[Boolean]()
    val crib = new ArrayBuilder[Int]()
    val cib = new ArrayBuilder[Int]()

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

  def unsafeOrdering(): UnsafeOrdering

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

  def equalModuloRequired(that: PType): Boolean = this == that.setRequired(required)

  final def orMissing(required2: Boolean): PType = {
    if (!required2)
      setRequired(false)
    else
      this
  }

  final def isOfType(t: PType): Boolean = this.virtualType == t.virtualType

  final def isPrimitive: Boolean =
    fundamentalType.isInstanceOf[PBoolean] || isNumeric

  final def isRealizable: Boolean = !isInstanceOf[PUnrealizable]

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
      case t: PArray => PCanonicalArray(t.elementType.deepInnerRequired(true), required)
      case t: PSet => PCanonicalSet(t.elementType.deepInnerRequired(true), required)
      case t: PDict => PCanonicalDict(t.keyType.deepInnerRequired(true), t.valueType.deepInnerRequired(true), required)
      case t: PStruct =>
        PCanonicalStruct(t.fields.map(f => PField(f.name, f.typ.deepInnerRequired(true), f.index)), required)
      case t: PCanonicalTuple =>
        PCanonicalTuple(t._types.map { f => f.copy(typ = f.typ.deepInnerRequired(true)) }, required)
      case t: PInterval =>
        PCanonicalInterval(t.pointType.deepInnerRequired(true), required)
      case t: PStream =>
        PCanonicalStream(t.elementType.deepInnerRequired(true), required)
      case t =>
        t.setRequired(required)
    }

  // Semantics: must be callable without requiredeness check: srcAddress must point to non-null value
  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long]

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_]

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_]): Code[_] =
    this.copyFromTypeAndStackValue(mb, region, srcPType, stackValue, false)

  protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long

  def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    // no requirement for requiredness
    // this can have more/less requiredness than srcPType
    // if value is not compatible with this, an exception will be thrown
    assert(virtualType == srcPType.virtualType)
    _copyFromAddress(region, srcPType, srcAddress, deepCopy)
  }

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit]
  def constructAtAddressFromValue(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, src: Code[_], deepCopy: Boolean): Code[Unit]
    = constructAtAddress(mb, addr, region, srcPType, asm4s.coerce[Long](src), deepCopy)

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit

  def deepRename(t: Type): PType = this

  def defaultValue: PCode = PCode(this, ir.defaultValue(this))

  def copyFromPValue(mb: EmitMethodBuilder[_], region: Value[Region], pv: PCode): PCode =
    PCode(this, copyFromTypeAndStackValue(mb, region, pv.pt, pv.code))

  final def typeCheck(a: Any): Boolean = a == null || _typeCheck(a)

  def _typeCheck(a: Any): Boolean = virtualType._typeCheck(a)

  def load(src: Code[Long]): PCode = PCode(this, Region.loadIRIntermediate(this)(src))

  def ti: TypeInfo[_] = typeToTypeInfo(this)

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(ti)

  def nCodes: Int = 1

  def codeReturnType(): TypeInfo[_] = ???

  def fromCodeTuple(ct: IndexedSeq[Code[_]]): PCode = {
    assert(ct.length == 1)
    PCode(this, ct(0))
  }
}
