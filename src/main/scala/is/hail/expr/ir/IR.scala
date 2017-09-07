package is.hail.expr.ir

import is.hail.asm4s.TypeInfo

import scala.language.existentials

sealed case class PrimitiveOp(x: Long)

sealed trait IR
sealed case class Null() extends IR
sealed case class I32(x: Int) extends IR
sealed case class I64(x: Long) extends IR
sealed case class F32(x: Float) extends IR
sealed case class F64(x: Double) extends IR
sealed case class True() extends IR
sealed case class False() extends IR
sealed case class If(cond: IR, cnsq: IR, altr: IR) extends IR
sealed case class Let(name: String, value: IR, typ: TypeInfo[_], body: IR) extends IR
sealed case class ApplyPrimitive(op: PrimitiveOp, args: Array[IR]) extends IR
sealed case class LazyApplyPrimitive(op: PrimitiveOp, args: Array[IR]) extends IR
sealed case class Lambda(name: String, body: IR) extends IR
sealed case class Ref(name: String) extends IR
sealed case class MakeArray(args: Array[IR], typ: TypeInfo[_]) extends IR
sealed case class ArrayRef(a: IR, i: IR, typ: TypeInfo[_]) extends IR
sealed case class MakeStruct(fields: Array[(String, TypeInfo[_], IR)]) extends IR
sealed case class GetField(o: IR, name: String) extends IR
sealed case class MapNull(name: String, value: IR, valueTyp: TypeInfo[_], body: IR) extends IR
sealed case class In(i: Int) extends IR
sealed case class Out(values: Array[IR]) extends IR

object IR {
  def usedInputs(ir: IR): Array[Int] = ir match {
    case Null() =>
      Array()
    case I32(x) =>
      Array()
    case I64(x) =>
      Array()
    case F32(x) =>
      Array()
    case F64(x) =>
      Array()
    case True() =>
      Array()
    case False() =>
      Array()
    case If(cond, cnsq, altr) =>
      usedInputs(cond) ++ usedInputs(cnsq) ++ usedInputs(altr)
    case Let(name, value, typ, body) =>
      usedInputs(value) ++ usedInputs(body)
    case ApplyPrimitive(op, args) =>
      args flatMap usedInputs
    case LazyApplyPrimitive(op, args) =>
      args flatMap usedInputs
    case Lambda(name, body) =>
      usedInputs(body)
    case Ref(name) =>
      Array()
    case MakeArray(args, typ) =>
      args flatMap usedInputs
    case ArrayRef(a, i, typ) =>
      usedInputs(a)
    case MakeStruct(fields) =>
      fields map (_._3) flatMap usedInputs
    case GetField(o, name) =>
      usedInputs(o)
    case MapNull(name, value, valueTyp, body) =>
      usedInputs(value) ++ usedInputs(body)
    case In(i) =>
      Array(i)
    case Out(values) =>
      values flatMap usedInputs
  }
}
