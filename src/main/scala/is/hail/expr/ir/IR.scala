package is.hail.expr.ir

import is.hail.expr.{NewAST, TBoolean, TFloat32, TFloat64, TInt32, TInt64, TStruct, TVoid, Type, TArray}

object IR {
  def seq(stmts: IR*)  = new Seq(stmts.toArray)
}

sealed trait IR extends NewAST {
  def typ: Type

  override def children: IndexedSeq[NewAST] = ???

  override def copy(newChildren: IndexedSeq[NewAST]): NewAST = ???
}
sealed case class NA(typ: Type) extends IR
sealed case class I32(x: Int) extends IR { val typ = TInt32 }
sealed case class I64(x: Long) extends IR { val typ = TInt64 }
sealed case class F32(x: Float) extends IR { val typ = TFloat32 }
sealed case class F64(x: Double) extends IR { val typ = TFloat64 }
sealed case class True() extends IR { val typ = TBoolean }
sealed case class False() extends IR { val typ = TBoolean }

sealed case class If(cond: IR, cnsq: IR, altr: IR, var typ: Type = null) extends IR
sealed case class MapNA(name: String, value: IR, body: IR, var typ: Type = null) extends IR

sealed case class Let(name: String, value: IR, body: IR, var typ: Type = null) extends IR
sealed case class Ref(name: String, var typ: Type = null) extends IR
sealed case class Set(name: String, v: IR) extends IR { val typ = TVoid }

sealed case class ApplyPrimitive(op: String, args: Array[IR], var typ: Type = null) extends IR
sealed case class LazyApplyPrimitive(op: String, args: Array[IR], var typ: Type = null) extends IR

sealed case class Lambda(name: String, paramTyp: Type, body: IR, var typ: Type = null) extends IR

sealed case class MakeArray(args: Array[IR], var typ: TArray = null) extends IR
sealed case class ArrayRef(a: IR, i: IR, var typ: Type = null) extends IR
sealed case class ArrayLen(a: IR) extends IR { val typ = TInt32 }
sealed case class For(value: String, idx: String, array: IR, body: IR) extends IR { val typ = TVoid }

sealed case class MakeStruct(fields: Array[(String, Type, IR)]) extends IR { val typ: TStruct = TStruct(fields.map(x => x._1 -> x._2):_*) }
sealed case class GetField(o: IR, name: String, var typ: Type = null) extends IR

sealed case class Seq(stmts: Array[IR], var typ: Type = null) extends IR {
  override def toString(): String = s"Seq(${stmts: IndexedSeq[IR]}, typ)"
}

sealed case class In(i: Int, var typ: Type = null) extends IR
sealed case class Out(v: IR) extends IR { val typ = TVoid }
