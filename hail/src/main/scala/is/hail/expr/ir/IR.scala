package is.hail.expr.ir

import is.hail.annotations.Annotation
import is.hail.expr.ir.functions._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{BufferSpec, CodecSpec2}
import is.hail.utils.{FastIndexedSeq, _}

import scala.language.existentials

sealed trait IR extends BaseIR {
  protected[ir] var _pType2: PType = null
  private var _pType: PType = null
  private var _typ: Type = null

  def pType = {
    if (_pType == null)
      _pType = PType.canonical(typ)

    _pType
  }

  def pType2 = {
    assert(_pType2 != null)

    _pType2
  }

  def typ: Type = {
    if (_typ == null)
      try {
        _typ = InferType(this)
      } catch {
        case e: Throwable => throw new RuntimeException(s"typ: inference failure: \n${ Pretty(this) }", e)
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
    if (_pType != null)
      cp._pType = _pType
    cp
  }

  def size: Int = 1 + children.map {
      case x: IR => x.size
      case _ => 0
    }.sum

  private[this] def _unwrap: IR => IR = {
    case node: ApplyIR => MapIR(_unwrap)(node.explicitNode)
    case node => MapIR(_unwrap)(node)
  }

  def unwrap: IR = _unwrap(this)
}

object Literal {
  def coerce(t: Type, x: Any): IR = {
    if (x == null)
      return NA(t)
    t match {
      case _: TInt32 => I32(x.asInstanceOf[Int])
      case _: TInt64 => I64(x.asInstanceOf[Long])
      case _: TFloat32 => F32(x.asInstanceOf[Float])
      case _: TFloat64 => F64(x.asInstanceOf[Double])
      case _: TBoolean => if (x.asInstanceOf[Boolean]) True() else False()
      case _: TString => Str(x.asInstanceOf[String])
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

object Coalesce {
  def unify(values: Seq[IR], unifyType: Option[Type] = None): Coalesce = {
    require(values.nonEmpty)
    val t1 = values.head.typ
    if (values.forall(_.typ == t1))
      Coalesce(values)
    else {
      val t = unifyType.getOrElse(t1.deepOptional())
      Coalesce(values.map(PruneDeadFields.upcast(_, t)))
    }
  }
}

final case class Coalesce(values: Seq[IR]) extends IR {
  require(values.nonEmpty)
}

object If {
  def unify(cond: IR, cnsq: IR, altr: IR, unifyType: Option[Type] = None): If = {
    if (cnsq.typ == altr.typ)
      If(cond, cnsq, altr)
    else {
      cnsq match {
        case NA(_) => If(cond, NA(altr.typ), altr)
        case Die(msg, _) => If(cond, Die(msg, altr.typ), altr)
        case Literal(_, value) if altr.typ.typeCheck(value) => If(cond, Literal(altr.typ, value), altr)
        case _ =>
          altr match {
            case NA(_) => If(cond, cnsq, NA(cnsq.typ))
            case Die(msg, _) => If(cond, cnsq, Die(msg, cnsq.typ))
            case Literal(_, value) if cnsq.typ.typeCheck(value)  => If(cond, cnsq, Literal(cnsq.typ, value))
            case _ =>
              val t = unifyType.getOrElse(cnsq.typ.deepOptional())
              If(cond,
                PruneDeadFields.upcast(cnsq, t),
                PruneDeadFields.upcast(altr, t))
          }
      }
    }
  }
}

final case class If(cond: IR, cnsq: IR, altr: IR) extends IR

final case class AggLet(name: String, value: IR, body: IR, isScan: Boolean) extends IR
final case class Let(name: String, value: IR, body: IR) extends IR
final case class Ref(name: String, var _typ: Type) extends IR

final case class RelationalLet(name: String, value: IR, body: IR) extends IR
final case class RelationalRef(name: String, _typ: Type) extends IR

final case class ApplyBinaryPrimOp(op: BinaryOp, l: IR, r: IR) extends IR
final case class ApplyUnaryPrimOp(op: UnaryOp, x: IR) extends IR
final case class ApplyComparisonOp(op: ComparisonOp[_], l: IR, r: IR) extends IR

object MakeArray {
  def unify(args: Seq[IR], typ: TArray = null): MakeArray = {
    assert(typ != null || args.nonEmpty)
    var t: TArray = typ
    if (t == null) {
      t = if (args.tail.forall(_.typ == args.head.typ)) {
        TArray(args.head.typ)
      } else TArray(args.head.typ.deepOptional())
    }
    assert(t.elementType.deepOptional() == t.elementType ||
      args.forall(a => a.typ == t.elementType),
      s"${ t.parsableString() }: ${ args.map(a => "\n    " + a.typ.parsableString()).mkString } ")

    MakeArray(args.map { arg =>
      if (arg.typ == t.elementType)
        arg
      else {
        val upcast = PruneDeadFields.upcast(arg, t.elementType)
        assert(upcast.typ == t.elementType)
        upcast
      }
    }, t)
  }
}

final case class MakeArray(args: Seq[IR], _typ: TArray) extends IR
final case class MakeStream(args: Seq[IR], _typ: TStream) extends IR
final case class ArrayRef(a: IR, i: IR) extends IR
final case class ArrayLen(a: IR) extends IR
final case class ArrayRange(start: IR, stop: IR, step: IR) extends IR
final case class StreamRange(start: IR, stop: IR, step: IR) extends IR


object ArraySort {
  def apply(a: IR, ascending: IR = True(), onKey: Boolean = false): ArraySort = {
    val l = genUID()
    val r = genUID()
    val atyp = coerce[TIterable](a.typ)
    val compare = if (onKey) {
      a.typ match {
        case atyp: TDict =>
          ApplyComparisonOp(Compare(atyp.keyType), GetField(Ref(l, atyp.elementType), "key"), GetField(Ref(r, atyp.elementType), "key"))
        case atyp: TStreamable if atyp.elementType.isInstanceOf[TStruct] =>
          val elt = coerce[TStruct](atyp.elementType)
          ApplyComparisonOp(Compare(elt.types(0)), GetField(Ref(l, elt), elt.fieldNames(0)), GetField(Ref(r, atyp.elementType), elt.fieldNames(0)))
        case atyp: TStreamable if atyp.elementType.isInstanceOf[TTuple] =>
          val elt = coerce[TTuple](atyp.elementType)
          ApplyComparisonOp(Compare(elt.types(0)), GetTupleElement(Ref(l, elt), elt.fields(0).index), GetTupleElement(Ref(r, atyp.elementType), elt.fields(0).index))
      }
    } else {
      ApplyComparisonOp(Compare(atyp.elementType), Ref(l, -atyp.elementType), Ref(r, -atyp.elementType))
    }

    ArraySort(a, l, r, If(ascending, compare < 0, compare > 0))
  }
}
final case class ArraySort(a: IR, left: String, right: String, compare: IR) extends IR
final case class ToSet(a: IR) extends IR
final case class ToDict(a: IR) extends IR
final case class ToArray(a: IR) extends IR
final case class ToStream(a: IR) extends IR

final case class LowerBoundOnOrderedCollection(orderedCollection: IR, elem: IR, onKey: Boolean) extends IR

final case class GroupByKey(collection: IR) extends IR

final case class ArrayMap(a: IR, name: String, body: IR) extends IR {
  override def typ: TStreamable = coerce[TStreamable](super.typ)
  def elementTyp: Type = typ.elementType
}
final case class ArrayFilter(a: IR, name: String, cond: IR) extends IR {
  override def typ: TStreamable = coerce[TStreamable](super.typ)
}
final case class ArrayFlatMap(a: IR, name: String, body: IR) extends IR {
  override def typ: TStreamable = coerce[TStreamable](super.typ)
}
final case class ArrayFold(a: IR, zero: IR, accumName: String, valueName: String, body: IR) extends IR

final case class ArrayScan(a: IR, zero: IR, accumName: String, valueName: String, body: IR) extends IR

final case class ArrayFor(a: IR, valueName: String, body: IR) extends IR

final case class ArrayAgg(a: IR, name: String, query: IR) extends IR
final case class ArrayAggScan(a: IR, name: String, query: IR) extends IR

final case class ArrayLeftJoinDistinct(left: IR, right: IR, l: String, r: String, keyF: IR, joinF: IR) extends IR

final case class MakeNDArray(data: IR, shape: IR, rowMajor: IR) extends IR

final case class NDArrayShape(nd: IR) extends IR

final case class NDArrayReshape(nd: IR, shape: IR) extends IR {
  require(shape.typ.asInstanceOf[TTuple].size > 0)
}

final case class NDArrayRef(nd: IR, idxs: IndexedSeq[IR]) extends IR
final case class NDArraySlice(nd: IR, slices: IR) extends IR

final case class NDArrayMap(nd: IR, valueName: String, body: IR) extends IR {
  override def typ: TNDArray = coerce[TNDArray](super.typ)
  def elementTyp: Type = typ.elementType
}

final case class NDArrayMap2(l: IR, r: IR, lName: String, rName: String, body: IR) extends IR {
  override def typ: TNDArray = coerce[TNDArray](super.typ)
  def elementTyp: Type = typ.elementType
}

final case class NDArrayReindex(nd: IR, indexExpr: IndexedSeq[Int]) extends IR
final case class NDArrayAgg(nd: IR, axes: IndexedSeq[Int]) extends IR
final case class NDArrayWrite(nd: IR, path: IR) extends IR

final case class NDArrayMatMul(l: IR, r: IR) extends IR

final case class AggFilter(cond: IR, aggIR: IR, isScan: Boolean) extends IR

final case class AggExplode(array: IR, name: String, aggBody: IR, isScan: Boolean) extends IR

final case class AggGroupBy(key: IR, aggIR: IR, isScan: Boolean) extends IR

final case class AggArrayPerElement(a: IR, elementName: String, indexName: String, aggBody: IR, knownLength: Option[IR], isScan: Boolean) extends IR

final case class ApplyAggOp(constructorArgs: IndexedSeq[IR], initOpArgs: Option[IndexedSeq[IR]], seqOpArgs: IndexedSeq[IR], aggSig: AggSignature) extends IR {
  assert(!(seqOpArgs ++ constructorArgs ++ initOpArgs.getOrElse(FastIndexedSeq.empty[IR])).exists(ContainsScan(_)))
  assert(constructorArgs.map(_.typ) == aggSig.constructorArgs)
  assert(initOpArgs.map(_.map(_.typ)) == aggSig.initOpArgs)

  def nSeqOpArgs = seqOpArgs.length

  def nConstructorArgs = constructorArgs.length

  def hasInitOp = initOpArgs.isDefined

  def op: AggOp = aggSig.op
}

final case class ApplyScanOp(constructorArgs: IndexedSeq[IR], initOpArgs: Option[IndexedSeq[IR]], seqOpArgs: IndexedSeq[IR], aggSig: AggSignature) extends IR {
  assert(!(seqOpArgs ++ constructorArgs ++ initOpArgs.getOrElse(FastIndexedSeq.empty[IR])).exists(ContainsAgg(_)))
  assert(constructorArgs.map(_.typ) == aggSig.constructorArgs)
  assert(initOpArgs.map(_.map(_.typ)) == aggSig.initOpArgs)

  def nSeqOpArgs = seqOpArgs.length

  def nConstructorArgs = constructorArgs.length

  def hasInitOp = initOpArgs.isDefined

  def op: AggOp = aggSig.op
}

final case class InitOp(i: IR, args: IndexedSeq[IR], aggSig: AggSignature) extends IR
final case class SeqOp(i: IR, args: IndexedSeq[IR], aggSig: AggSignature) extends IR

final case class InitOp2(i: Int, args: IndexedSeq[IR], aggSig: AggSignature2) extends IR
final case class SeqOp2(i: Int, args: IndexedSeq[IR], aggSig: AggSignature2) extends IR
final case class CombOp2(i1: Int, i2: Int, aggSig: AggSignature2) extends IR
final case class ResultOp2(startIdx: Int, aggSigs: IndexedSeq[AggSignature2]) extends IR

final case class SerializeAggs(startIdx: Int, serializedIdx: Int, spec: BufferSpec, aggSigs: IndexedSeq[AggSignature2]) extends IR
final case class DeserializeAggs(startIdx: Int, serializedIdx: Int, spec: BufferSpec, aggSigs: IndexedSeq[AggSignature2]) extends IR

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

final case class In(i: Int, _typ: Type) extends IR

// FIXME: should be type any
object Die {
  def apply(message: String, typ: Type): Die = Die(Str(message), typ)
}

final case class Die(message: IR, _typ: Type) extends IR

final case class ApplyIR(function: String, args: Seq[IR]) extends IR {
  var conversion: Seq[IR] => IR = _

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

final case class Uniroot(argname: String, function: IR, min: IR, max: IR) extends IR

final case class TableCount(child: TableIR) extends IR
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

final case class BlockMatrixWrite(child: BlockMatrixIR, writer: BlockMatrixWriter) extends IR

final case class BlockMatrixMultiWrite(blockMatrices: IndexedSeq[BlockMatrixIR], writer: BlockMatrixMultiWriter) extends IR

final case class CollectDistributedArray(contexts: IR, globals: IR, cname: String, gname: String, body: IR) extends IR
final case class ReadPartition(path: IR, spec: CodecSpec2, rowType: TStruct) extends IR

class PrimitiveIR(val self: IR) extends AnyVal {
  def +(other: IR): IR = ApplyBinaryPrimOp(Add(), self, other)
  def -(other: IR): IR = ApplyBinaryPrimOp(Subtract(), self, other)
  def *(other: IR): IR = ApplyBinaryPrimOp(Multiply(), self, other)
  def /(other: IR): IR = ApplyBinaryPrimOp(FloatingPointDivide(), self, other)
  def floorDiv(other: IR): IR = ApplyBinaryPrimOp(RoundToNegInfDivide(), self, other)

  def &&(other: IR): IR = invoke("&&", TBoolean(), self, other)
  def ||(other: IR): IR = invoke("||", TBoolean(), self, other)

  def toI: IR = Cast(self, TInt32())
  def toL: IR = Cast(self, TInt64())
  def toF: IR = Cast(self, TFloat32())
  def toD: IR = Cast(self, TFloat64())

  def unary_-(): IR = ApplyUnaryPrimOp(Negate(), self)
  def unary_!(): IR = ApplyUnaryPrimOp(Bang(), self)

  def ceq(other: IR): IR = ApplyComparisonOp(EQWithNA(self.typ, other.typ), self, other)
  def cne(other: IR): IR = ApplyComparisonOp(NEQWithNA(self.typ, other.typ), self, other)
  def <(other: IR): IR = ApplyComparisonOp(LT(self.typ, other.typ), self, other)
  def >(other: IR): IR = ApplyComparisonOp(GT(self.typ, other.typ), self, other)
  def <=(other: IR): IR = ApplyComparisonOp(LTEQ(self.typ, other.typ), self, other)
  def >=(other: IR): IR = ApplyComparisonOp(GTEQ(self.typ, other.typ), self, other)
}
