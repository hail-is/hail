package is.hail.expr.ir

import is.hail.expr.{BaseIR, TBoolean, TFloat32, TFloat64, TInt32, TInt64, TStruct, TVoid, Type, TArray}

sealed trait IR extends BaseIR {
  def typ: Type

  override def children: IndexedSeq[BaseIR] = ???

  override def copy(newChildren: IndexedSeq[BaseIR]): BaseIR = ???
}
sealed case class I32(x: Int) extends IR { val typ = TInt32 }
sealed case class I64(x: Long) extends IR { val typ = TInt64 }
sealed case class F32(x: Float) extends IR { val typ = TFloat32 }
sealed case class F64(x: Double) extends IR { val typ = TFloat64 }
sealed case class True() extends IR { val typ = TBoolean }
sealed case class False() extends IR { val typ = TBoolean }

sealed case class NA(typ: Type) extends IR
sealed case class MapNA(name: String, value: IR, body: IR, var typ: Type = null) extends IR
sealed case class IsNA(value: IR) extends IR { val typ = TBoolean }

sealed case class If(cond: IR, cnsq: IR, altr: IR, var typ: Type = null) extends IR

sealed case class Let(name: String, value: IR, body: IR, var typ: Type = null) extends IR
sealed case class Ref(name: String, var typ: Type = null) extends IR

sealed case class ApplyPrimitive(op: String, args: Array[IR], var typ: Type = null) extends IR {
  override def toString(): String = s"ApplyPrimitive($op, ${args: IndexedSeq[IR]}, $typ)"
}
sealed case class LazyApplyPrimitive(op: String, args: Array[IR], var typ: Type = null) extends IR {
  override def toString(): String = s"LazyApplyPrimitive($op, ${args: IndexedSeq[IR]}, $typ)"
}

sealed case class Lambda(names: Array[(String, Type)], body: IR, var typ: Type = null) extends IR {
  override def toString(): String = s"Lambda(${names: IndexedSeq[(String, Type)]}, $body, $typ)"
}

sealed case class MakeArray(args: Array[IR], var typ: TArray = null) extends IR {
  override def toString(): String = s"MakeArray(${args: IndexedSeq[IR]}, $typ)"
}
sealed case class MakeArrayN(len: IR, elementType: Type) extends IR { def typ: TArray = TArray(elementType) }
sealed case class ArrayRef(a: IR, i: IR, var typ: Type = null) extends IR
sealed case class ArrayMissingnessRef(a: IR, i: IR) extends IR { val typ: Type = TBoolean }
sealed case class ArrayLen(a: IR) extends IR { val typ = TInt32 }
sealed case class ArrayMap(a: IR, lam: IR, var elementTyp: Type = null) extends IR { def typ: TArray = TArray(elementTyp) }
sealed case class ArrayFold(a: IR, zero: IR, lam: IR, var typ: Type = null) extends IR

sealed case class MakeStruct(fields: Array[(String, Type, IR)]) extends IR {
  val typ: TStruct = TStruct(fields.map(x => x._1 -> x._2):_*)
  override def toString(): String =
    s"MakeStruct(${fields: IndexedSeq[(String, Type, IR)]})"
}
sealed case class GetField(o: IR, name: String, var typ: Type = null) extends IR
sealed case class GetFieldMissingness(o: IR, name: String) extends IR { val typ: Type = TBoolean }

sealed case class In(i: Int, val typ: Type) extends IR
sealed case class InMissingness(i: Int) extends IR { val typ: Type = TBoolean }
sealed case class Out(v: IR) extends IR { val typ = TVoid }
// FIXME: should be type any
sealed case class Die(message: String) extends IR { val typ = TVoid }
