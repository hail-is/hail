package is.hail.expr

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.types._

import scala.language.implicitConversions

package object ir {
  var uidCounter: Long = 0

  def genUID(): String = {
    val uid = s"__iruid_$uidCounter"
    uidCounter += 1
    uid
  }

  def typeToTypeInfo(t: Type): TypeInfo[_] = t.fundamentalType match {
    case _: TInt32 => typeInfo[Int]
    case _: TInt64 => typeInfo[Long]
    case _: TFloat32 => typeInfo[Float]
    case _: TFloat64 => typeInfo[Double]
    case _: TBoolean => typeInfo[Boolean]
    case _: TBinary => typeInfo[Long]
    case _: TArray => typeInfo[Long]
    case _: TBaseStruct => typeInfo[Long]
    case TVoid => typeInfo[Unit]
    case _ => throw new RuntimeException(s"unsupported type found, $t")
  }

  def defaultValue(t: Type): Code[_] = typeToTypeInfo(t) match {
    case UnitInfo => Code._empty[Unit]
    case BooleanInfo => false
    case IntInfo => 0
    case LongInfo => 0L
    case FloatInfo => 0.0f
    case DoubleInfo => 0.0
    case ti => throw new RuntimeException(s"unsupported type found: $t whose type info is $ti")
  }

  // Build consistent expression for a filter-condition with keep polarity,
  // using Let to manage missing-ness.
  def filterPredicateWithKeep(irPred: ir.IR, keep: Boolean): ir.IR = {
    val pred = genUID()
    ir.Let(pred,
      if (keep) irPred else ir.ApplyUnaryPrimOp(ir.Bang(), irPred),
      ir.If(ir.IsNA(ir.Ref(pred, TBoolean())),
        ir.False(),
        ir.Ref(pred, TBoolean())))
  }

  private[ir] def coerce[T](c: Code[_]): Code[T] = asm4s.coerce(c)

  private[ir] def coerce[T](lr: Settable[_]): Settable[T] = lr.asInstanceOf[Settable[T]]

  private[ir] def coerce[T](ti: TypeInfo[_]): TypeInfo[T] = ti.asInstanceOf[TypeInfo[T]]

  private[ir] def coerce[T <: Type](x: Type): T = types.coerce[T](x)

  def invoke(name: String, args: IR*): IR =
    IRFunctionRegistry.lookupConversion(name, args.map(_.typ)).get(args)

  implicit def irToPrimitiveIR(ir: IR): PrimitiveIR = new PrimitiveIR(ir)

  implicit def intToIR(i: Int): IR = I32(i)
  implicit def longToIR(l: Long): IR = I64(l)
  implicit def floatToIR(f: Float): IR = F32(f)
  implicit def doubleToIR(d: Double): IR = F64(d)
  implicit def booleanToIR(b: Boolean): IR = if (b) True() else False()
}
