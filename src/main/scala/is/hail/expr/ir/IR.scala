package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.expr.{BaseIR, MatrixIR, TableIR}
import is.hail.expr.ir.functions.{IRFunction, IRFunctionRegistry, IRFunctionWithMissingness, IRFunctionWithoutMissingness}
import is.hail.utils.ExportType

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
      case _ => throw new RuntimeException(s"Unsupported literal type: $t")
    }
  }
}

trait InferIR extends IR {
  var _typ: Type = null

  def typ: Type = {
    if (_typ == null)
      _typ = Infer(this)
    _typ
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

final case class If(cond: IR, cnsq: IR, altr: IR) extends InferIR

final case class Let(name: String, value: IR, body: IR) extends InferIR
final case class Ref(name: String, var typ: Type) extends IR

final case class ApplyBinaryPrimOp(op: BinaryOp, l: IR, r: IR) extends InferIR
final case class ApplyUnaryPrimOp(op: UnaryOp, x: IR) extends InferIR

final case class MakeArray(args: Seq[IR], typ: TArray) extends IR
final case class ArrayRef(a: IR, i: IR) extends InferIR
final case class ArrayLen(a: IR) extends IR { val typ = TInt32() }
final case class ArrayRange(start: IR, stop: IR, step: IR) extends IR { val typ: TArray = TArray(TInt32()) }
final case class ArrayMap(a: IR, name: String, body: IR) extends InferIR {
  override def typ: TArray = coerce[TArray](super.typ)
  def elementTyp: Type = typ.elementType
}
final case class ArrayFilter(a: IR, name: String, cond: IR) extends InferIR {
  override def typ: TArray = super.typ.asInstanceOf[TArray]
}
final case class ArrayFlatMap(a: IR, name: String, body: IR) extends InferIR {
  override def typ: TArray = coerce[TArray](super.typ)
}
final case class ArrayFold(a: IR, zero: IR, accumName: String, valueName: String, body: IR) extends InferIR

final case class AggIn(var typ: TAggregable) extends IR
final case class AggMap(a: IR, name: String, body: IR) extends InferIR
final case class AggFilter(a: IR, name: String, body: IR) extends InferIR
final case class AggFlatMap(a: IR, name: String, body: IR) extends InferIR
final case class ApplyAggOp(a: IR, op: AggOp, args: Seq[IR]) extends InferIR {
  def inputType: Type = coerce[TAggregable](a.typ).elementType
}

final case class MakeStruct(fields: Seq[(String, IR)]) extends InferIR

final case class InsertFields(old: IR, fields: Seq[(String, IR)]) extends InferIR {
  override def typ: TStruct = coerce[TStruct](super.typ)
}

final case class GetField(o: IR, name: String) extends InferIR

final case class MakeTuple(types: Seq[IR]) extends InferIR
final case class GetTupleElement(o: IR, idx: Int) extends InferIR

final case class In(i: Int, typ: Type) extends IR
// FIXME: should be type any
final case class Die(message: String) extends IR { val typ = TVoid }

final case class Apply(function: String, args: Seq[IR]) extends IR {
  val implementation: IRFunctionWithoutMissingness =
    IRFunctionRegistry.lookupFunction(function, args.map(_.typ)).get.asInstanceOf[IRFunctionWithoutMissingness]

  def typ: Type = {
    // convert all arg types before unifying
    val argTypes = args.map(_.typ)
    implementation.argTypes.foreach(_.clear())
    argTypes.zip(implementation.argTypes).foreach { case (i, j) =>
      val u = j.unify(i)
      assert(u)
    }
    implementation.returnType.subst()
  }

  def isDeterministic: Boolean = implementation.isDeterministic
}

final case class ApplySpecial(function: String, args: Seq[IR]) extends IR {
  val implementation: IRFunctionWithMissingness =
    IRFunctionRegistry.lookupFunction(function, args.map(_.typ)).get.asInstanceOf[IRFunctionWithMissingness]

  def typ: Type = {
    val argTypes = args.map(_.typ)
    implementation.argTypes.foreach(_.clear())
    argTypes.zip(implementation.argTypes).foreach { case (i, j) =>
      val u = j.unify(i)
      assert(u)
    }
    implementation.returnType.subst()
  }

  def isDeterministic: Boolean = implementation.isDeterministic
}

final case class TableCount(child: TableIR) extends IR { val typ: Type = TInt64() }
final case class TableAggregate(child: TableIR, query: IR) extends InferIR
final case class TableWrite(
  child: TableIR,
  path: String,
  overwrite: Boolean = true,
  codecSpecJSONStr: String = null) extends IR {
  val typ: Type = TVoid
}
final case class TableExport(
  child: TableIR,
  path: String,
  typesFile: String = null,
  header: Boolean = true,
  exportType: Int = ExportType.CONCATENATED) extends IR {
  val typ: Type = TVoid
}

final case class MatrixWrite(
  child: MatrixIR,
  path: String,
  overwrite: Boolean = true,
  codecSpecJSONStr: String = null) extends IR {
  val typ: Type = TVoid
}
