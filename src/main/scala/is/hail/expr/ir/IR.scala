package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.{BaseIR, MatrixIR, TableIR}
import is.hail.expr.ir.functions.{IRFunction, IRFunctionWithMissingness, IRFunctionWithoutMissingness}

sealed trait IR extends BaseIR {
  def typ: Type

  override def children: IndexedSeq[BaseIR] =
    Children(this)

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR =
    Copy(this, newChildren)

  def size: Int = 1 + children.map {
      case x: IR => x.size
      case _ => 0
    }.sum
}

object Literal {
  def apply(x: Any, t: Type): IR = {
    if (x == null)
      return NA(t)
    t match {
      case _: TInt32 => I32(x.asInstanceOf[Int])
      case _: TInt64 => I64(x.asInstanceOf[Long])
      case _: TFloat32 => F32(x.asInstanceOf[Float])
      case _: TFloat64 => F64(x.asInstanceOf[Double])
      case _: TBoolean => if (x.asInstanceOf[Boolean]) True() else False()
      case _: TString => Str(x.asInstanceOf[String])
      case _ => throw new RuntimeException("Unsupported literal type")
    }
  }
}

final case class I32(x: Int) extends IR { val typ = TInt32() }
final case class I64(x: Long) extends IR { val typ = TInt64() }
final case class F32(x: Float) extends IR { val typ = TFloat32() }
final case class F64(x: Double) extends IR { val typ = TFloat64() }
final case class Str(x: String) extends IR { val typ = TString() }
final case class True() extends IR { val typ = TBoolean() }
final case class False() extends IR { val typ = TBoolean() }

final case class Cast(v: IR, typ: Type) extends IR

final case class NA(typ: Type) extends IR
final case class IsNA(value: IR) extends IR { val typ = TBoolean() }

final case class If(cond: IR, cnsq: IR, altr: IR, var typ: Type = null) extends IR

final case class Let(name: String, value: IR, body: IR, var typ: Type = null) extends IR
final case class Ref(name: String, var typ: Type = null) extends IR

final case class ApplyBinaryPrimOp(op: BinaryOp, l: IR, r: IR, var typ: Type = null) extends IR
final case class ApplyUnaryPrimOp(op: UnaryOp, x: IR, var typ: Type = null) extends IR

final case class MakeArray(args: Seq[IR], var typ: TArray = null) extends IR
final case class ArrayRef(a: IR, i: IR, var typ: Type = null) extends IR
final case class ArrayLen(a: IR) extends IR { val typ = TInt32() }
final case class ArrayRange(start: IR, stop: IR, step: IR) extends IR { val typ: TArray = TArray(TInt32()) }

final case class ArraySort(a: IR) extends IR { def typ: TArray = coerce[TArray](a.typ) }
final case class Set(a: IR) extends IR { def typ: TSet = TSet(coerce[TArray](a.typ).elementType) }
final case class Dict(a: IR) extends IR {
  def typ: TDict = {
    val etype = coerce[TBaseStruct](coerce[TArray](a.typ).elementType)
    TDict(etype.types(0), etype.types(1))
  }
}

final case class ArrayMap(a: IR, name: String, body: IR, var elementTyp: Type = null) extends IR { def typ: TArray = TArray(elementTyp) }
final case class ArrayFilter(a: IR, name: String, cond: IR) extends IR { def typ: TArray = coerce[TArray](a.typ) }
final case class ArrayFlatMap(a: IR, name: String, body: IR) extends IR { def typ: TArray = coerce[TArray](body.typ) }
final case class ArrayFold(a: IR, zero: IR, accumName: String, valueName: String, body: IR, var typ: Type = null) extends IR

final case class AggIn(var typ: TAggregable = null) extends IR
final case class AggMap(a: IR, name: String, body: IR, var typ: TAggregable = null) extends IR
final case class AggFilter(a: IR, name: String, body: IR, var typ: TAggregable = null) extends IR
final case class AggFlatMap(a: IR, name: String, body: IR, var typ: TAggregable = null) extends IR
final case class ApplyAggOp(a: IR, op: AggOp, args: Seq[IR], var typ: Type = null) extends IR {
  def inputType: Type = coerce[TAggregable](a.typ).elementType
}

final case class MakeStruct(fields: Seq[(String, IR)], var typ: TStruct = null) extends IR
final case class InsertFields(old: IR, fields: Seq[(String, IR)], var typ: TStruct = null) extends IR

final case class GetField(o: IR, name: String, var typ: Type = null) extends IR

final case class MakeTuple(types: Seq[IR], var typ: TTuple = null) extends IR
final case class GetTupleElement(o: IR, idx: Int, var typ: Type = null) extends IR

final case class In(i: Int, var typ: Type) extends IR
// FIXME: should be type any
final case class Die(message: String) extends IR { val typ = TVoid }

final case class Apply(function: String, args: Seq[IR], var implementation: IRFunctionWithoutMissingness = null) extends IR { def typ = implementation.returnType }
final case class ApplySpecial(function: String, args: Seq[IR], var implementation: IRFunctionWithMissingness = null) extends IR { def typ = implementation.returnType }

final case class TableCount(child: TableIR) extends IR { val typ: Type = TInt64() }
final case class TableAggregate(child: TableIR, query: IR, var typ: Type = null) extends IR
final case class TableWrite(
  child: TableIR,
  path: String,
  overwrite: Boolean = true,
  codecSpecJSONStr: String = null) extends IR {
  val typ: Type = TVoid
}

final case class MatrixWrite(
  child: MatrixIR,
  path: String,
  overwrite: Boolean = true,
  codecSpecJSONStr: String = null) extends IR {
  val typ: Type = TVoid
}
