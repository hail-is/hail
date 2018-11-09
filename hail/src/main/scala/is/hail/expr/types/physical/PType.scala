package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.check.{Arbitrary, Gen}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.virtual.Type
import is.hail.expr.{JSONAnnotationImpex, Parser, SparkAnnotationImpex}
import is.hail.utils
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.types.DataType
import org.json4s.JValue

import scala.reflect.ClassTag

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

  val optionalComplex: Gen[PType] = genComplexType(false)

  val requiredComplex: Gen[PType] = genComplexType(true)

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
    for (fields <- genFields(required, genFieldType)) yield {
      val t = PStruct(fields)
      if (required)
        (+t).asInstanceOf[PStruct]
      else
        t
    }
  }

  def preGenTuple(required: Boolean, genFieldType: Gen[PType]): Gen[PTuple] = {
    for (fields <- genFields(required, genFieldType)) yield {
      val t = PTuple(fields.map(_.typ))
      if (required)
        (+t).asInstanceOf[PTuple]
      else
        t
    }
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
}

abstract class PType extends BaseType with Serializable {
  self =>

  def virtualType: Type

  def children: Seq[PType] = FastSeq()

  def clear(): Unit = children.foreach(_.clear())

  def unify(concrete: PType): Boolean = {
    this.isOfType(concrete)
  }

  def isBound: Boolean = children.forall(_.isBound)

  def subst(): PType = this.setRequired(false)

  def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = ???

  def unsafeOrdering(): UnsafeOrdering = unsafeOrdering(false)

  def unsafeOrdering(rightType: PType, missingGreatest: Boolean): UnsafeOrdering = {
    require(this.isOfType(rightType))
    unsafeOrdering(missingGreatest)
  }

  def unsafeOrdering(rightType: PType): UnsafeOrdering = unsafeOrdering(rightType, false)

  def unsafeInsert(typeToInsert: PType, path: List[String]): (PType, UnsafeInserter) =
    PStruct.empty().unsafeInsert(typeToInsert, path)

  def insert(signature: PType, fields: String*): (PType, Inserter) = insert(signature, fields.toList)

  def insert(signature: PType, path: List[String]): (PType, Inserter) = {
    if (path.nonEmpty)
      PStruct.empty().insert(signature, path)
    else
      (signature, (a, toIns) => toIns)
  }

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (required)
      sb.append("+")
    _pretty(sb, indent, compact)
  }

  def _toPretty: String

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append(_toPretty)
  }

  def fieldOption(fields: String*): Option[PField] = fieldOption(fields.toList)

  def fieldOption(path: List[String]): Option[PField] =
    None

  def isRealizable: Boolean = children.forall(_.isRealizable)

  def scalaClassTag: ClassTag[_ <: AnyRef]

  def canCompare(other: PType): Boolean = this == other

  def codeOrdering(mb: EmitMethodBuilder): CodeOrdering = codeOrdering(mb, this)

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering

  def byteSize: Long = 1

  def alignment: Long = byteSize

  /*  Fundamental types are types that can be handled natively by RegionValueBuilder: primitive
      types, Array and Struct. */
  def fundamentalType: PType = this

  def required: Boolean

  final def unary_+(): PType = setRequired(true)

  final def unary_-(): PType = setRequired(false)

  final def setRequired(required: Boolean): PType = {
    this match {
      case PBinary(_) => PBinary(required)
      case PBoolean(_) => PBoolean(required)
      case PInt32(_) => PInt32(required)
      case PInt64(_) => PInt64(required)
      case PFloat32(_) => PFloat32(required)
      case PFloat64(_) => PFloat64(required)
      case PString(_) => PString(required)
      case PCall(_) => PCall(required)
      case t: PArray => t.copy(required = required)
      case t: PSet => t.copy(required = required)
      case t: PDict => t.copy(required = required)
      case t: PLocus => t.copy(required = required)
      case t: PInterval => t.copy(required = required)
      case t: PStruct => t.copy(required = required)
      case t: PTuple => t.copy(required = required)
    }
  }

  final def isOfType(t: PType): Boolean = {
    this match {
      case PBinary(_) => t == PBinaryOptional || t == PBinaryRequired
      case PBoolean(_) => t == PBooleanOptional || t == PBooleanRequired
      case PInt32(_) => t == PInt32Optional || t == PInt32Required
      case PInt64(_) => t == PInt64Optional || t == PInt64Required
      case PFloat32(_) => t == PFloat32Optional || t == PFloat32Required
      case PFloat64(_) => t == PFloat64Optional || t == PFloat64Required
      case PString(_) => t == PStringOptional || t == PStringRequired
      case PCall(_) => t == PCallOptional || t == PCallRequired
      case t2: PLocus => t.isInstanceOf[PLocus] && t.asInstanceOf[PLocus].rg == t2.rg
      case t2: PInterval => t.isInstanceOf[PInterval] && t.asInstanceOf[PInterval].pointType.isOfType(t2.pointType)
      case t2: PStruct =>
        t.isInstanceOf[PStruct] &&
          t.asInstanceOf[PStruct].size == t2.size &&
          t.asInstanceOf[PStruct].fields.zip(t2.fields).forall { case (f1: PField, f2: PField) => f1.typ.isOfType(f2.typ) && f1.name == f2.name }
      case t2: PTuple =>
        t.isInstanceOf[PTuple] &&
          t.asInstanceOf[PTuple].size == t2.size &&
          t.asInstanceOf[PTuple].types.zip(t2.types).forall { case (typ1, typ2) => typ1.isOfType(typ2) }
      case t2: PArray => t.isInstanceOf[PArray] && t.asInstanceOf[PArray].elementType.isOfType(t2.elementType)
      case t2: PSet => t.isInstanceOf[PSet] && t.asInstanceOf[PSet].elementType.isOfType(t2.elementType)
      case t2: PDict => t.isInstanceOf[PDict] && t.asInstanceOf[PDict].keyType.isOfType(t2.keyType) && t.asInstanceOf[PDict].valueType.isOfType(t2.valueType)
    }
  }

  def deepOptional(): PType =
    this match {
      case t: PArray => PArray(t.elementType.deepOptional())
      case t: PSet => PSet(t.elementType.deepOptional())
      case t: PDict => PDict(t.keyType.deepOptional(), t.valueType.deepOptional())
      case t: PStruct =>
        PStruct(t.fields.map(f => PField(f.name, f.typ.deepOptional(), f.index)))
      case t: PTuple =>
        PTuple(t.types.map(_.deepOptional()))
      case t =>
        t.setRequired(false)
    }

  def structOptional(): PType =
    this match {
      case t: PStruct =>
        PStruct(t.fields.map(f => PField(f.name, f.typ.deepOptional(), f.index)))
      case t =>
        t.setRequired(false)
    }
}
