package is.hail.expr

import is.hail.asm4s
import is.hail.asm4s._
import is.hail.annotations.RegionValue
import is.hail.asm4s.joinpoint.Ctrl
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.types.{coerce => tycoerce, _}
import is.hail.types.physical.PType
import is.hail.types.virtual._
import is.hail.utils._

import scala.language.implicitConversions

import java.util.UUID

package object ir {
  type TokenIterator = BufferedIterator[Token]

  var uidCounter: Long = 0

  def genUID(): String = {
    val uid = s"__iruid_$uidCounter"
    uidCounter += 1
    uid
  }

  def uuid4(): String = UUID.randomUUID().toString

  def genSym(base: String): Sym = Sym.gen(base)

  def typeToTypeInfo(t: PType): TypeInfo[_] = typeToTypeInfo(t.virtualType)

  def typeToTypeInfo(t: Type): TypeInfo[_] = t.fundamentalType match {
    case TInt32 => typeInfo[Int]
    case TInt64 => typeInfo[Long]
    case TFloat32 => typeInfo[Float]
    case TFloat64 => typeInfo[Double]
    case TBoolean => typeInfo[Boolean]
    case TBinary => typeInfo[Long]
    case _: TShuffle => typeInfo[Long]
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

  private[ir] def coerce[T <: Type](x: Type): T = tycoerce[T](x)

  private[ir] def coerce[T <: PType](x: PType): T = tycoerce[T](x)

  private[ir] def coerce[T <: BaseTypeWithRequiredness](x: BaseTypeWithRequiredness): T = tycoerce[T](x)

  def invoke(name: String, rt: Type, typeArgs: Array[Type], args: IR*): IR = IRFunctionRegistry.lookupUnseeded(name, rt, typeArgs, args.map(_.typ)) match {
    case Some(f) => f(typeArgs, args)
    case None => fatal(s"no conversion found for $name(${typeArgs.mkString(", ")}, ${args.map(_.typ).mkString(", ")}) => $rt")
  }

  def invoke(name: String, rt: Type, args: IR*): IR =
    invoke(name, rt, Array.empty[Type], args:_*)

  def invokeSeeded(name: String, seed: Long, rt: Type, args: IR*): IR = IRFunctionRegistry.lookupSeeded(name, seed, rt, args.map(_.typ)) match {
    case Some(f) => f(args)
    case None => fatal(s"no seeded function found for $name(${args.map(_.typ).mkString(", ")}) => $rt")
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

  def bindIRs(values: IR*)(body: Seq[Ref] => IR): IR = {
    val valuesArray = values.toArray
    val refs = values.map(v => Ref(genUID(), v.typ))
    values.indices.foldLeft(body(refs)) { case (acc, i) => Let(refs(i).name, valuesArray(i), acc) }
  }

  def bindIR(v: IR)(body: Ref => IR): IR = {
    val ref = Ref(genUID(), v.typ)
    Let(ref.name, v, body(ref))
  }

  def maxIR(a: IR, b: IR): IR = {
    If(a > b, a, b)
  }

  def minIR(a: IR, b: IR): IR = {
    If(a < b, a, b)
  }

  def forIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), coerce[TStream](stream.typ).elementType)
    StreamFor(stream, ref.name, f(ref))
  }

  def mapIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), coerce[TStream](stream.typ).elementType)
    StreamMap(stream, ref.name, f(ref))
  }

  def flatMapIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), coerce[TStream](stream.typ).elementType)
    StreamFlatMap(stream, ref.name, f(ref))
  }

  def flatten(stream: IR): IR = flatMapIR(stream) { elt =>
      if (elt.typ.isInstanceOf[TStream]) elt else ToStream(elt)
    }

  def foldIR(stream: IR, zero: IR)(f: (Ref, Ref) => IR): IR = {
    val elt = Ref(genUID(), coerce[TStream](stream.typ).elementType)
    val accum = Ref(genUID(), zero.typ)
    StreamFold(stream, zero, accum.name, elt.name, f(accum, elt))
  }

  def streamSumIR(stream: IR): IR = {
    foldIR(stream, 0){ case (accum, elt) => accum + elt}
  }

  def streamForceCount(stream: IR): IR =
    streamSumIR(mapIR(stream)(_ => I32(1)))

  def rangeIR(n: IR): IR = StreamRange(0, n, 1)

  def rangeIR(start: IR, stop: IR): IR = StreamRange(start, stop, 1)

  def insertIR(old: IR, fields: (String, IR)*): InsertFields = InsertFields(old, fields)
  def selectIR(old: IR, fields: String*): SelectFields = SelectFields(old, fields)

  def zip2(s1: IR, s2: IR, behavior: ArrayZipBehavior.ArrayZipBehavior)(f: (Ref, Ref) => IR): IR = {
    val r1 = Ref(genUID(), coerce[TStream](s1.typ).elementType)
    val r2 = Ref(genUID(), coerce[TStream](s2.typ).elementType)
    StreamZip(FastSeq(s1, s2), FastSeq(r1.name, r2.name), f(r1, r2), behavior)
  }

  def zipIR(ss: IndexedSeq[IR], behavior: ArrayZipBehavior.ArrayZipBehavior)(f: IndexedSeq[Ref] => IR): IR = {
    val refs = ss.map(s => Ref(genUID(), coerce[TStream](s.typ).elementType))
    StreamZip(ss, refs.map(_.name), f(refs), behavior)
  }

  def makestruct(fields: (String, IR)*): MakeStruct = MakeStruct(fields)

  implicit def toRichIndexedSeqEmitSettable(s: IndexedSeq[EmitSettable]): RichIndexedSeqEmitSettable = new RichIndexedSeqEmitSettable(s)

  implicit def emitValueToCode(ev: EmitValue): EmitCode = ev.get

  implicit def toCodeParamType(ti: TypeInfo[_]): CodeParamType = CodeParamType(ti)

  implicit def toEmitParamType(pt: PType): EmitParamType = EmitParamType(pt)

  implicit def toCodeParam(c: Code[_]): CodeParam = CodeParam(c)

  implicit def valueToCodeParam(v: Value[_]): CodeParam = CodeParam(v)

  implicit def toEmitParam(ec: EmitCode): EmitParam = EmitParam(ec)

  implicit def emitValueToEmitParam(ev: EmitValue): EmitParam = EmitParam(ev)
}
