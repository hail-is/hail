package is.hail.expr

import is.hail.asm4s._
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.types.physical.stypes.SValue
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.utils._

import java.util.UUID
import scala.language.implicitConversions

package object ir {
  type TokenIterator = BufferedIterator[Token]
  type IEmitCode = IEmitCodeGen[SValue]

  var uidCounter: Long = 0

  def genUID(): String = {
    val uid = s"__iruid_$uidCounter"
    uidCounter += 1
    uid
  }

  def uuid4(): String = UUID.randomUUID().toString

  def genSym(base: String): Sym = Sym.gen(base)

  // Build consistent expression for a filter-condition with keep polarity,
  // using Let to manage missing-ness.
  def filterPredicateWithKeep(irPred: ir.IR, keep: Boolean): ir.IR =
    bindIR(if (keep) irPred else ir.ApplyUnaryPrimOp(ir.Bang, irPred)) { pred =>
      ir.If(ir.IsNA(pred), ir.False(), pred)
    }

  def invoke(name: String, rt: Type, typeArgs: Array[Type], errorID: Int, args: IR*): IR = IRFunctionRegistry.lookupUnseeded(name, rt, typeArgs, args.map(_.typ)) match {
    case Some(f) => f(typeArgs, args, errorID)
    case None => fatal(s"no conversion found for $name(${typeArgs.mkString(", ")}, ${args.map(_.typ).mkString(", ")}) => $rt")
  }
  def invoke(name: String, rt: Type, typeArgs: Array[Type], args: IR*): IR =
    invoke(name, rt, typeArgs, ErrorIDs.NO_ERROR, args:_*)

  def invoke(name: String, rt: Type, args: IR*): IR =
    invoke(name, rt, Array.empty[Type], ErrorIDs.NO_ERROR, args:_*)

  def invoke(name: String, rt: Type, errorID: Int, args: IR*): IR =
    invoke(name, rt, Array.empty[Type], errorID, args:_*)

  def invokeSeeded(name: String, staticUID: Long, rt: Type, rngState: IR, args: IR*): IR =
    IRFunctionRegistry.lookupSeeded(name, staticUID, rt, args.map(_.typ)) match {
      case Some(f) => f(args, rngState)
      case None => fatal(s"no seeded function found for $name(${args.map(_.typ).mkString(", ")}) => $rt")
    }

  implicit def irToPrimitiveIR(ir: IR): PrimitiveIR = new PrimitiveIR(ir)

  implicit def intToIR(i: Int): I32 = I32(i)

  implicit def longToIR(l: Long): I64 = I64(l)

  implicit def floatToIR(f: Float): F32 = F32(f)

  implicit def doubleToIR(d: Double): F64 = F64(d)

  implicit def booleanToIR(b: Boolean): TrivialIR = if (b) True() else False()

  def zero(t: Type): IR = t match {
    case TInt32 => I32(0)
    case TInt64 => I64(0L)
    case TFloat32 => F32(0f)
    case TFloat64 => F64(0d)
  }

  def bindIRs(values: IR*)(body: Seq[Ref] => IR): IR = {
    val bindings = values.toFastSeq.map(genUID() -> _)
    Let(bindings, body(bindings.map(b => Ref(b._1, b._2.typ))))
  }

  def bindIR(v: IR)(body: Ref => IR): IR =
    bindIRs(v) { case Seq(ref) => body(ref) }

  def iota(start: IR, step: IR): IR = StreamIota(start, step)

  def dropWhile(v: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](v.typ).elementType)
    StreamDropWhile(v, ref.name, f(ref))
  }

  def takeWhile(v: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](v.typ).elementType)
    StreamTakeWhile(v, ref.name, f(ref))
  }

  def maxIR(a: IR, b: IR): IR = {
    If(a > b, a, b)
  }

  def minIR(a: IR, b: IR): IR = {
    If(a < b, a, b)
  }

  def streamAggIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](stream.typ).elementType)
    StreamAgg(stream, ref.name, f(ref))
  }

  def forIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](stream.typ).elementType)
    StreamFor(stream, ref.name, f(ref))
  }

  def filterIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](stream.typ).elementType)
    StreamFilter(stream, ref.name, f(ref))
  }

  def mapIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](stream.typ).elementType)
    StreamMap(stream, ref.name, f(ref))
  }

  def flatMapIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(genUID(), tcoerce[TStream](stream.typ).elementType)
    StreamFlatMap(stream, ref.name, f(ref))
  }

  def flatten(stream: IR): IR = flatMapIR(if (stream.typ.isInstanceOf[TStream]) stream else ToStream(stream)) { elt =>
    if (elt.typ.isInstanceOf[TStream]) elt else ToStream(elt)
  }

  def foldIR(stream: IR, zero: IR)(f: (Ref, Ref) => IR): IR = {
    val elt = Ref(genUID(), tcoerce[TStream](stream.typ).elementType)
    val accum = Ref(genUID(), zero.typ)
    StreamFold(stream, zero, accum.name, elt.name, f(accum, elt))
  }

  def sortIR(stream: IR)(f: (Ref, Ref) => IR): IR = {
    val t = tcoerce[TStream](stream.typ).elementType
    val l = Ref(genUID(), t)
    val r = Ref(genUID(), t)
    ArraySort(stream, l.name, r.name, f(l, r))
  }

  def sliceArrayIR(arrayIR: IR, startIR: IR, stopIR: IR): IR = {
    ArraySlice(arrayIR, startIR, Some(stopIR))
  }

  def joinIR(left: IR, right: IR, lkey: IndexedSeq[String], rkey: IndexedSeq[String], joinType: String, requiresMemoryManagement: Boolean)(f: (Ref, Ref) => IR): IR = {
    val lRef = Ref(genUID(), left.typ.asInstanceOf[TStream].elementType)
    val rRef = Ref(genUID(), right.typ.asInstanceOf[TStream].elementType)
    StreamJoin(left, right, lkey, rkey, lRef.name, rRef.name, f(lRef, rRef), joinType, requiresMemoryManagement)
  }

  def joinRightDistinctIR(left: IR, right: IR, lkey: IndexedSeq[String], rkey: IndexedSeq[String], joinType: String)(f: (Ref, Ref) => IR): IR = {
    val lRef = Ref(genUID(), left.typ.asInstanceOf[TStream].elementType)
    val rRef = Ref(genUID(), right.typ.asInstanceOf[TStream].elementType)
    StreamJoinRightDistinct(left, right, lkey, rkey, lRef.name, rRef.name, f(lRef, rRef), joinType)
  }

  def streamSumIR(stream: IR): IR = {
    foldIR(stream, 0){ case (accum, elt) => accum + elt}
  }

  def streamForceCount(stream: IR): IR =
    streamSumIR(mapIR(stream)(_ => I32(1)))

  def rangeIR(n: IR): IR = StreamRange(0, n, 1)

  def rangeIR(start: IR, stop: IR): IR = StreamRange(start, stop, 1)

  def insertIR(old: IR, fields: (String, IR)*): InsertFields = InsertFields(old, fields.toArray[(String, IR)])
  def selectIR(old: IR, fields: String*): SelectFields = SelectFields(old, fields.toArray[String])

  def zip2(s1: IR, s2: IR, behavior: ArrayZipBehavior.ArrayZipBehavior)(f: (Ref, Ref) => IR): IR = {
    val r1 = Ref(genUID(), tcoerce[TStream](s1.typ).elementType)
    val r2 = Ref(genUID(), tcoerce[TStream](s2.typ).elementType)
    StreamZip(FastSeq(s1, s2), FastSeq(r1.name, r2.name), f(r1, r2), behavior)
  }

  def zipWithIndex(s: IR): IR = {
    val r1 = Ref(genUID(), tcoerce[TStream](s.typ).elementType)
    val r2 = Ref(genUID(), TInt32)
    StreamZip(
      FastSeq(s, StreamIota(I32(0), I32(1))),
      FastSeq(r1.name, r2.name),
      MakeStruct(FastSeq(("elt", r1), ("idx", r2))),
      ArrayZipBehavior.TakeMinLength
    )
  }

  def zipIR(ss: IndexedSeq[IR], behavior: ArrayZipBehavior.ArrayZipBehavior, errorId: Int = ErrorIDs.NO_ERROR)(f: IndexedSeq[Ref] => IR): IR = {
    val refs = ss.map(s => Ref(genUID(), tcoerce[TStream](s.typ).elementType))
    StreamZip(ss, refs.map(_.name), f(refs), behavior, errorId)
  }

  def makestruct(fields: (String, IR)*): MakeStruct = MakeStruct(fields.toArray[(String, IR)])
  def maketuple(fields: IR*): MakeTuple = MakeTuple(fields.toArray.zipWithIndex.map { case (field, idx) => (idx, field) })

  def aggBindIR(v: IR, isScan: Boolean = false)(body: Ref => IR): IR = {
    val ref = Ref(genUID(), v.typ)
    AggLet(ref.name, v, body(ref), isScan = isScan)
  }

  def aggExplodeIR(v: IR, isScan: Boolean = false)(body: Ref => IR): AggExplode = {
    val r = Ref(genUID(), v.typ.asInstanceOf[TIterable].elementType)
    AggExplode(v, r.name, body(r), isScan)
  }

  def aggFoldIR(zero: IR, element: IR)(seqOp: (Ref, IR) => IR)(combOp: (Ref, Ref) => IR) : AggFold = {
    val accum1 = Ref(genUID(), zero.typ)
    val accum2 = Ref(genUID(), zero.typ)
    AggFold(zero, seqOp(accum1, element), combOp(accum1, accum2), accum1.name, accum2.name, false)
  }

  def cdaIR(contexts: IR, globals: IR, staticID: String, dynamicID: IR = NA(TString))(body: (Ref, Ref) => IR): CollectDistributedArray = {
    val contextRef = Ref(genUID(), contexts.typ.asInstanceOf[TStream].elementType)
    val globalRef = Ref(genUID(), globals.typ)

    CollectDistributedArray(contexts, globals, contextRef.name, globalRef.name, body(contextRef, globalRef), dynamicID, staticID, None)
  }

  def strConcat(irs: AnyRef*): IR = {
    assert(irs.nonEmpty)
    var s: IR = null
    irs.foreach { xAny =>
      val x = xAny match {
        case x: IR => x
        case x: String => Str(x)
      }

      if (s == null)
        s = x
      else {
        val xstr = if (x.typ == TString)
          x
        else
          invoke("str", TString, x)
        s = invoke("concat", TString, s, xstr)
      }
    }
    s
  }

  def logIR(result: IR, messages: AnyRef*): IR = ConsoleLog(strConcat(messages: _*), result)

  implicit def toRichIndexedSeqEmitSettable(s: IndexedSeq[EmitSettable]): RichIndexedSeqEmitSettable = new RichIndexedSeqEmitSettable(s)

  implicit def emitValueToCode(ev: EmitValue): EmitCode = ev.load

  implicit def toCodeParamType(ti: TypeInfo[_]): CodeParamType = CodeParamType(ti)

  implicit def toCodeParam(c: Value[_]): CodeParam = CodeParam(c)

  implicit def sValueToSCodeParam(sv: SValue): SCodeParam = SCodeParam(sv)

  implicit def toEmitParam(ec: EmitCode): EmitParam = EmitParam(ec)

  implicit def emitValueToEmitParam(ev: EmitValue): EmitParam = EmitParam(ev)
}
