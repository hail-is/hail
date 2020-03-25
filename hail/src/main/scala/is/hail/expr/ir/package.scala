package is.hail.expr

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.annotations.RegionValue
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.types._
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual._
import is.hail.utils._

import scala.language.implicitConversions

package object ir {
  type TokenIterator = BufferedIterator[Token]

  var uidCounter: Long = 0

  def genUID(): String = {
    val uid = s"__iruid_$uidCounter"
    uidCounter += 1
    uid
  }

  def genSym(base: String): Sym = Sym.gen(base)

  def typeToTypeInfo(t: PType): TypeInfo[_] = typeToTypeInfo(t.virtualType)

  def typeToTypeInfo(t: Type): TypeInfo[_] = t.fundamentalType match {
    case TInt32 => typeInfo[Int]
    case TInt64 => typeInfo[Long]
    case TFloat32 => typeInfo[Float]
    case TFloat64 => typeInfo[Double]
    case TBoolean => typeInfo[Boolean]
    case TBinary => typeInfo[Long]
    case _: TArray => typeInfo[Long]
    case _: TBaseStruct => typeInfo[Long]
    case _: TStream => classInfo[Iterator[RegionValue]]
    case TVoid => typeInfo[Unit]
    case _ => throw new RuntimeException(s"unsupported type found, $t")
  }

  def defaultValue(t: PType): Code[_] = defaultValue(t.virtualType)

  def defaultValue(t: Type): Code[_] = defaultValue(typeToTypeInfo(t))

  def defaultValue(ti: TypeInfo[_]): Code[_] = ti match {
    case UnitInfo => Code._empty
    case BooleanInfo => false
    case IntInfo => 0
    case LongInfo => 0L
    case FloatInfo => 0.0f
    case DoubleInfo => 0.0
    case _: ClassInfo[_] => Code._null
    case ti => throw new RuntimeException(s"unsupported type found: $ti")
  }

  // Build consistent expression for a filter-condition with keep polarity,
  // using Let to manage missing-ness.
  def filterPredicateWithKeep(irPred: ir.IR, keep: Boolean): ir.IR = {
    val pred = genUID()
    ir.Let(pred,
      if (keep) irPred else ir.ApplyUnaryPrimOp(ir.Bang(), irPred),
      ir.If(ir.IsNA(ir.Ref(pred, TBoolean)),
        ir.False(),
        ir.Ref(pred, TBoolean)))
  }

  private[ir] def coerce[T](c: Code[_]): Code[T] = asm4s.coerce(c)

  private[ir] def coerce[T](c: Value[_]): Value[T] = asm4s.coerce(c)

  private[ir] def coerce[T](lr: Settable[_]): Settable[T] = lr.asInstanceOf[Settable[T]]

  private[ir] def coerce[T](ti: TypeInfo[_]): TypeInfo[T] = ti.asInstanceOf[TypeInfo[T]]

  private[ir] def coerce[T <: Type](x: Type): T = types.coerce[T](x)

  private[ir] def coerce[T <: PType](x: PType): T = types.coerce[T](x)

  def invoke(name: String, rt: Type, args: IR*): IR = IRFunctionRegistry.lookupConversion(name, rt, args.map(_.typ)) match {
    case Some(f) => f(args)
    case None => fatal(s"no conversion found for $name(${args.map(_.typ).mkString(", ")}) => $rt")
  }


  implicit def irToPrimitiveIR(ir: IR): PrimitiveIR = new PrimitiveIR(ir)

  implicit def intToIR(i: Int): IR = I32(i)

  implicit def longToIR(l: Long): IR = I64(l)

  implicit def floatToIR(f: Float): IR = F32(f)

  implicit def doubleToIR(d: Double): IR = F64(d)

  implicit def booleanToIR(b: Boolean): IR = if (b) True() else False()

  def zero(t: Type): IR = t match {
    case TInt32 => I32(0)
    case TInt64 => I64(0L)
    case TFloat32 => F32(0f)
    case TFloat64 => F64(0d)
  }


  def mapIR(stream: IR)(f: IR => IR): IR = {
    val ref = Ref(genUID(), coerce[TStream](stream.typ).elementType)
    StreamMap(stream, ref.name, f(ref))
  }

  def rangeIR(n: IR): IR = StreamRange(0, n, 1)

  def rangeIR(start: IR, stop: IR): IR = StreamRange(start, stop, 1)

  implicit def toRichIndexedSeqEmitSettable(s: IndexedSeq[EmitSettable]): RichIndexedSeqEmitSettable = new RichIndexedSeqEmitSettable(s)

  implicit def emitValueToCode(ev: EmitValue): EmitCode = ev.get
}
