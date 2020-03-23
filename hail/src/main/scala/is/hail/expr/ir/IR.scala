package is.hail.expr.ir

import is.hail.annotations.Annotation
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.ir.functions._
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec}
import is.hail.utils.{FastIndexedSeq, _}

import scala.language.existentials

sealed trait IR extends BaseIR {
  protected[ir] var _pType: PType = null
  private var _typ: Type = null

  def pType = {
    assert(_pType != null)

    _pType
  }

  def typ: Type = {
    if (_typ == null)
      try {
        _typ = InferType(this)
      } catch {
        case e: Throwable => throw new RuntimeException(s"typ: inference failure", e)
      }
    _typ
  }

  lazy val children: IndexedSeq[BaseIR] =
    Children(this)

  override def copy(newChildren: IndexedSeq[BaseIR]): IR =
    Copy(this, newChildren)

  override def deepCopy(): this.type = {

    val cp = super.deepCopy()
    if (_typ != null)
      cp._typ = _typ
    cp
  }

  lazy val size: Int = 1 + children.map {
      case x: IR => x.size
      case _ => 0
    }.sum

  private[this] def _unwrap: IR => IR = {
    case node: ApplyIR => MapIR(_unwrap)(node.explicitNode)
    case node => MapIR(_unwrap)(node)
  }

  def unwrap: IR = _unwrap(this)
}

sealed trait TypedIR[T <: Type, P <: PType] extends IR {
  override def typ: T = coerce[T](super.typ)
  override def pType: P = coerce[P](super.pType)
}

object Literal {
  def coerce(t: Type, x: Any): IR = {
    if (x == null)
      return NA(t)
    t match {
      case TInt32 => I32(x.asInstanceOf[Int])
      case TInt64 => I64(x.asInstanceOf[Long])
      case TFloat32 => F32(x.asInstanceOf[Float])
      case TFloat64 => F64(x.asInstanceOf[Double])
      case TBoolean => if (x.asInstanceOf[Boolean]) True() else False()
      case TString => Str(x.asInstanceOf[String])
      case _ => Literal(t, x)
    }
  }
}

final case class Literal(_typ: Type, value: Annotation) extends IR {
  require(!CanEmit(_typ))
  require(value != null)
}

final case class I32(x: Int) extends IR
final case class I64(x: Long) extends IR
final case class F32(x: Float) extends IR
final case class F64(x: Double) extends IR
final case class Str(x: String) extends IR
final case class True() extends IR
final case class False() extends IR
final case class Void() extends IR

final case class Cast(v: IR, _typ: Type) extends IR
final case class CastRename(v: IR, _typ: Type) extends IR

final case class NA(_typ: Type) extends IR
final case class IsNA(value: IR) extends IR

final case class Coalesce(values: Seq[IR]) extends IR {
  require(values.nonEmpty)
}

final case class If(cond: IR, cnsq: IR, altr: IR) extends IR

final case class AggLet(name: String, value: IR, body: IR, isScan: Boolean) extends IR
final case class Let(name: String, value: IR, body: IR) extends IR
final case class Ref(name: String, var _typ: Type) extends IR


// Recur can't exist outside of loop
// Loops can be nested, but we can't call outer loops in terms of inner loops so there can only be one loop "active" in a given context
final case class TailLoop(name: String, params: IndexedSeq[(String, IR)], body: IR) extends IR
final case class Recur(name: String, args: IndexedSeq[IR], _typ: Type) extends IR

final case class RelationalLet(name: String, value: IR, body: IR) extends IR
final case class RelationalRef(name: String, _typ: Type) extends IR

final case class ApplyBinaryPrimOp(op: BinaryOp, l: IR, r: IR) extends IR
final case class ApplyUnaryPrimOp(op: UnaryOp, x: IR) extends IR
final case class ApplyComparisonOp(op: ComparisonOp[_], l: IR, r: IR) extends IR

object MakeArray {
  def unify(args: Seq[IR], requestedType: TArray = null): MakeArray = {
    assert(requestedType != null || args.nonEmpty)

    if(args.nonEmpty)
      if (args.forall(_.typ == args.head.typ))
        return MakeArray(args, TArray(args.head.typ))

    MakeArray(args.map { arg =>
      val upcast = PruneDeadFields.upcast(arg, requestedType.elementType)
      assert(upcast.typ == requestedType.elementType)
      upcast
    }, requestedType)
  }
}

final case class MakeArray(args: Seq[IR], _typ: TArray) extends IR

object MakeStream {
  def unify(args: Seq[IR], requestedType: TStream = null): MakeStream = {
    assert(requestedType != null || args.nonEmpty)

    if (args.nonEmpty)
      if (args.forall(_.typ == args.head.typ))
        return MakeStream(args, TStream(args.head.typ))

    MakeStream(args.map { arg =>
      val upcast = PruneDeadFields.upcast(arg, requestedType.elementType)
      assert(upcast.typ == requestedType.elementType)
      upcast
    }, requestedType)
  }
}

final case class MakeStream(args: Seq[IR], _typ: TStream) extends IR

object ArrayRef {
  def apply(a: IR, i: IR): ArrayRef = ArrayRef(a, i, Str(""))
}

final case class ArrayRef(a: IR, i: IR, msg: IR) extends IR
final case class ArrayLen(a: IR) extends IR
final case class ArrayZeros(length: IR) extends IR
final case class StreamRange(start: IR, stop: IR, step: IR) extends IR

object ArraySort {
  def apply(a: IR, ascending: IR = True(), onKey: Boolean = false): ArraySort = {
    val l = genUID()
    val r = genUID()
    val atyp = coerce[TStream](a.typ)
    val compare = if (onKey) {
      val elementType = atyp.elementType.asInstanceOf[TBaseStruct]
      elementType match {
        case t: TStruct =>
          val elt = coerce[TStruct](atyp.elementType)
          ApplyComparisonOp(Compare(elt.types(0)), GetField(Ref(l, elt), elt.fieldNames(0)), GetField(Ref(r, atyp.elementType), elt.fieldNames(0)))
        case t: TTuple =>
          val elt = coerce[TTuple](atyp.elementType)
          ApplyComparisonOp(Compare(elt.types(0)), GetTupleElement(Ref(l, elt), elt.fields(0).index), GetTupleElement(Ref(r, atyp.elementType), elt.fields(0).index))
      }
    } else {
      ApplyComparisonOp(Compare(atyp.elementType), Ref(l, atyp.elementType), Ref(r, atyp.elementType))
    }

    ArraySort(a, l, r, If(ascending, compare < 0, compare > 0))
  }
}

final case class ArraySort(a: IR, left: String, right: String, compare: IR) extends IR
final case class ToSet(a: IR) extends IR
final case class ToDict(a: IR) extends IR
final case class ToArray(a: IR) extends IR
final case class CastToArray(a: IR) extends IR
final case class ToStream(a: IR) extends IR

final case class LowerBoundOnOrderedCollection(orderedCollection: IR, elem: IR, onKey: Boolean) extends IR

final case class GroupByKey(collection: IR) extends IR

final case class StreamMap(a: IR, name: String, body: IR) extends IR {
  override def typ: TStream = coerce[TStream](super.typ)
  def elementTyp: Type = typ.elementType
}

object ArrayZipBehavior extends Enumeration {
  type ArrayZipBehavior = Value
  val AssumeSameLength: Value = Value(0)
  val AssertSameLength: Value = Value(1)
  val TakeMinLength: Value = Value(2)
  val ExtendNA: Value = Value(3)
}

final case class StreamZip(as: IndexedSeq[IR], names: IndexedSeq[String], body: IR, behavior: ArrayZipBehavior) extends IR {
  override def typ: TStream = coerce[TStream](super.typ)
}
final case class StreamFilter(a: IR, name: String, cond: IR) extends IR {
  override def typ: TStream = coerce[TStream](super.typ)
}
final case class StreamFlatMap(a: IR, name: String, body: IR) extends IR {
  override def typ: TStream = coerce[TStream](super.typ)
}
final case class StreamFold(a: IR, zero: IR, accumName: String, valueName: String, body: IR) extends IR

object StreamFold2 {
  def apply(a: StreamFold): StreamFold2 = {
    StreamFold2(a.a, FastIndexedSeq((a.accumName, a.zero)), a.valueName, FastSeq(a.body), Ref(a.accumName, a.zero.typ))
  }
}

final case class StreamFold2(a: IR, accum: IndexedSeq[(String, IR)], valueName: String, seq: IndexedSeq[IR], result: IR) extends IR {
  assert(accum.length == seq.length)
  var accPTypes: IndexedSeq[PType] = null
}

final case class StreamScan(a: IR, zero: IR, accumName: String, valueName: String, body: IR) extends IR {
  var accPType: PType = null
}

final case class StreamFor(a: IR, valueName: String, body: IR) extends IR

final case class StreamAgg(a: IR, name: String, query: IR) extends IR
final case class StreamAggScan(a: IR, name: String, query: IR) extends IR

trait InferredPhysicalAggSignature {
  // will be filled in by InferPType in subsequent PR
  var physicalSignatures: Array[AggStatePhysicalSignature] = _

  def signature: IndexedSeq[AggStateSignature]
}
final case class RunAgg(body: IR, result: IR, signature: IndexedSeq[AggStateSignature]) extends IR with InferredPhysicalAggSignature
final case class RunAggScan(array: IR, name: String, init: IR, seqs: IR, result: IR, signature: IndexedSeq[AggStateSignature]) extends IR with InferredPhysicalAggSignature

final case class StreamLeftJoinDistinct(left: IR, right: IR, l: String, r: String, keyF: IR, joinF: IR) extends IR

sealed trait NDArrayIR extends TypedIR[TNDArray, PNDArray] {
  def elementTyp: Type = typ.elementType
}

object MakeNDArray {
  def fill(elt: IR, shape: IndexedSeq[Long], rowMajor: IR): MakeNDArray =
    MakeNDArray(
      ToArray(StreamMap(StreamRange(0, shape.product, 1), genUID(), elt)),
      MakeTuple.ordered(shape.map(I64)), rowMajor)
}

final case class MakeNDArray(data: IR, shape: IR, rowMajor: IR) extends NDArrayIR

final case class NDArrayShape(nd: IR) extends IR

final case class NDArrayReshape(nd: IR, shape: IR) extends NDArrayIR

final case class NDArrayConcat(nds: IR, axis: Int) extends NDArrayIR

final case class NDArrayRef(nd: IR, idxs: IndexedSeq[IR]) extends IR
final case class NDArraySlice(nd: IR, slices: IR) extends NDArrayIR
final case class NDArrayFilter(nd: IR, keep: IndexedSeq[IR]) extends NDArrayIR

final case class NDArrayMap(nd: IR, valueName: String, body: IR) extends NDArrayIR
final case class NDArrayMap2(l: IR, r: IR, lName: String, rName: String, body: IR) extends NDArrayIR

final case class NDArrayReindex(nd: IR, indexExpr: IndexedSeq[Int]) extends NDArrayIR
final case class NDArrayAgg(nd: IR, axes: IndexedSeq[Int]) extends IR
final case class NDArrayWrite(nd: IR, path: IR) extends IR

final case class NDArrayMatMul(l: IR, r: IR) extends NDArrayIR

final case class NDArrayQR(nd: IR, mode: String) extends IR

final case class AggFilter(cond: IR, aggIR: IR, isScan: Boolean) extends IR

final case class AggExplode(array: IR, name: String, aggBody: IR, isScan: Boolean) extends IR

final case class AggGroupBy(key: IR, aggIR: IR, isScan: Boolean) extends IR

final case class AggArrayPerElement(a: IR, elementName: String, indexName: String, aggBody: IR, knownLength: Option[IR], isScan: Boolean) extends IR

final case class ApplyAggOp(initOpArgs: IndexedSeq[IR], seqOpArgs: IndexedSeq[IR], aggSig: AggSignature) extends IR {

  def nSeqOpArgs = seqOpArgs.length

  def nInitArgs = initOpArgs.length

  def op: AggOp = aggSig.op
}

final case class ApplyScanOp(initOpArgs: IndexedSeq[IR], seqOpArgs: IndexedSeq[IR], aggSig: AggSignature) extends IR {

  def nSeqOpArgs = seqOpArgs.length

  def nInitArgs = initOpArgs.length

  def op: AggOp = aggSig.op
}

object InitOp {
  def apply(i: Int, args: IndexedSeq[IR], aggSig: AggSignature): InitOp = InitOp(i, args, AggStateSignature(aggSig), aggSig.op)
}
final case class InitOp(i: Int, args: IndexedSeq[IR], aggSig: AggStateSignature, op: AggOp) extends IR

object SeqOp {
  def apply(i: Int, args: IndexedSeq[IR], aggSig: AggSignature): SeqOp = SeqOp(i, args, AggStateSignature(aggSig), aggSig.op)
}
final case class SeqOp(i: Int, args: IndexedSeq[IR], aggSig: AggStateSignature, op: AggOp) extends IR
final case class CombOp(i1: Int, i2: Int, aggSig: AggStateSignature) extends IR
final case class ResultOp(startIdx: Int, aggSigs: IndexedSeq[AggStateSignature]) extends IR
final case class CombOpValue(i: Int, value: IR, aggSig: AggStateSignature) extends IR
final case class AggStateValue(i: Int, aggSig: AggStateSignature) extends IR

final case class SerializeAggs(startIdx: Int, serializedIdx: Int, spec: BufferSpec, aggSigs: IndexedSeq[AggStateSignature]) extends IR
final case class DeserializeAggs(startIdx: Int, serializedIdx: Int, spec: BufferSpec, aggSigs: IndexedSeq[AggStateSignature]) extends IR

final case class Begin(xs: IndexedSeq[IR]) extends IR
final case class MakeStruct(fields: Seq[(String, IR)]) extends IR
final case class SelectFields(old: IR, fields: Seq[String]) extends IR

object InsertFields {
  def apply(old: IR, fields: Seq[(String, IR)]): InsertFields = InsertFields(old, fields, None)
}
final case class InsertFields(old: IR, fields: Seq[(String, IR)], fieldOrder: Option[IndexedSeq[String]]) extends IR {

  override def typ: TStruct = coerce[TStruct](super.typ)

  override def pType: PStruct = coerce[PStruct](super.pType)
}

object GetFieldByIdx {
  def apply(s: IR, field: Int): IR = {
    (s.typ: @unchecked) match {
      case t: TStruct => GetField(s, t.fieldNames(field))
      case _: TTuple => GetTupleElement(s, field)
    }
  }
}

final case class GetField(o: IR, name: String) extends IR

object MakeTuple {
  def ordered(types: Seq[IR]): MakeTuple = MakeTuple(types.iterator.zipWithIndex.map { case (ir, i) => (i, ir)}.toFastIndexedSeq)
}

final case class MakeTuple(fields: Seq[(Int, IR)]) extends IR
final case class GetTupleElement(o: IR, idx: Int) extends IR

object In {
  def apply(i: Int, typ: Type): In = In(i, PType.canonical(typ))
}

final case class In(i: Int, _typ: PType) extends IR

// FIXME: should be type any
object Die {
  def apply(message: String, typ: Type): Die = Die(Str(message), typ)
}

final case class Die(message: IR, _typ: Type) extends IR

final case class ApplyIR(function: String, args: Seq[IR]) extends IR {
  var conversion: Seq[IR] => IR = _
  var inline: Boolean = _

  private lazy val refs = args.map(a => Ref(genUID(), a.typ)).toArray
  lazy val body: IR = conversion(refs).deepCopy()

  lazy val explicitNode: IR = {
    // foldRight because arg1 should be at the top so it is evaluated first
    refs.zip(args).foldRight(body) { case ((ref, arg), bodyIR) => Let(ref.name, arg, bodyIR) }
  }
}

sealed abstract class AbstractApplyNode[F <: IRFunction] extends IR {
  def function: String
  def args: Seq[IR]
  def returnType: Type
  def argTypes: Seq[Type] = args.map(_.typ)
  lazy val implementation: F = IRFunctionRegistry.lookupFunction(function, returnType, argTypes)
    .getOrElse(throw new RuntimeException(s"no function match for $function: ${ argTypes.map(_.parsableString()).mkString(", ") }"))
      .asInstanceOf[F]
}

final case class Apply(function: String, args: Seq[IR], returnType: Type) extends AbstractApplyNode[IRFunctionWithoutMissingness]

final case class ApplySeeded(function: String, args: Seq[IR], seed: Long, returnType: Type) extends AbstractApplyNode[SeededIRFunction]

final case class ApplySpecial(function: String, args: Seq[IR], returnType: Type) extends AbstractApplyNode[IRFunctionWithMissingness]

final case class LiftMeOut(child: IR) extends IR
final case class TableCount(child: TableIR) extends IR
final case class MatrixCount(child: MatrixIR) extends IR
final case class TableAggregate(child: TableIR, query: IR) extends IR
final case class MatrixAggregate(child: MatrixIR, query: IR) extends IR

final case class TableWrite(child: TableIR, writer: TableWriter) extends IR

final case class TableMultiWrite(_children: IndexedSeq[TableIR], writer: WrappedMatrixNativeMultiWriter) extends IR {
  private val t = _children.head.typ
  require(_children.forall(_.typ == t))
}

final case class TableGetGlobals(child: TableIR) extends IR
final case class TableCollect(child: TableIR) extends IR

final case class MatrixWrite(child: MatrixIR, writer: MatrixWriter) extends IR

final case class MatrixMultiWrite(_children: IndexedSeq[MatrixIR], writer: MatrixNativeMultiWriter) extends IR {
  private val t = _children.head.typ
  require(_children.forall(_.typ == t))
}

final case class TableToValueApply(child: TableIR, function: TableToValueFunction) extends IR
final case class MatrixToValueApply(child: MatrixIR, function: MatrixToValueFunction) extends IR
final case class BlockMatrixToValueApply(child: BlockMatrixIR, function: BlockMatrixToValueFunction) extends IR

final case class BlockMatrixCollect(child: BlockMatrixIR) extends NDArrayIR

final case class BlockMatrixWrite(child: BlockMatrixIR, writer: BlockMatrixWriter) extends IR

final case class BlockMatrixMultiWrite(blockMatrices: IndexedSeq[BlockMatrixIR], writer: BlockMatrixMultiWriter) extends IR

final case class CollectDistributedArray(contexts: IR, globals: IR, cname: String, gname: String, body: IR) extends IR

final case class ReadPartition(path: IR, spec: AbstractTypedCodecSpec, rowType: TStruct) extends IR

final case class ReadValue(path: IR, spec: AbstractTypedCodecSpec, requestedType: Type) extends IR
final case class WriteValue(value: IR, pathPrefix: IR, spec: AbstractTypedCodecSpec) extends IR

final case class UnpersistBlockMatrix(child: BlockMatrixIR) extends IR

class PrimitiveIR(val self: IR) extends AnyVal {
  def +(other: IR): IR = ApplyBinaryPrimOp(Add(), self, other)
  def -(other: IR): IR = ApplyBinaryPrimOp(Subtract(), self, other)
  def *(other: IR): IR = ApplyBinaryPrimOp(Multiply(), self, other)
  def /(other: IR): IR = ApplyBinaryPrimOp(FloatingPointDivide(), self, other)
  def floorDiv(other: IR): IR = ApplyBinaryPrimOp(RoundToNegInfDivide(), self, other)

  def &&(other: IR): IR = invoke("land", TBoolean, self, other)
  def ||(other: IR): IR = invoke("lor", TBoolean, self, other)

  def toI: IR = Cast(self, TInt32)
  def toL: IR = Cast(self, TInt64)
  def toF: IR = Cast(self, TFloat32)
  def toD: IR = Cast(self, TFloat64)

  def unary_-(): IR = ApplyUnaryPrimOp(Negate(), self)
  def unary_!(): IR = ApplyUnaryPrimOp(Bang(), self)

  def ceq(other: IR): IR = ApplyComparisonOp(EQWithNA(self.typ, other.typ), self, other)
  def cne(other: IR): IR = ApplyComparisonOp(NEQWithNA(self.typ, other.typ), self, other)
  def <(other: IR): IR = ApplyComparisonOp(LT(self.typ, other.typ), self, other)
  def >(other: IR): IR = ApplyComparisonOp(GT(self.typ, other.typ), self, other)
  def <=(other: IR): IR = ApplyComparisonOp(LTEQ(self.typ, other.typ), self, other)
  def >=(other: IR): IR = ApplyComparisonOp(GTEQ(self.typ, other.typ), self, other)
}
