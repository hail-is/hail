package is.hail.expr

import is.hail.asm4s._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.rvd.RVDPartitioner
import is.hail.types.physical.stypes.SValue
import is.hail.types.tcoerce
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils._

import scala.collection.BufferedIterator

import java.util.UUID

package object ir {
  type TokenIterator = BufferedIterator[Token]
  type IEmitCode = IEmitCodeGen[SValue]

  var uidCounter: Long = 0

  def genUID(): String = {
    val uid = s"__iruid_$uidCounter"
    uidCounter += 1
    uid
  }

  def freshName(): Name = Name(genUID())

  def uuid4(): String = UUID.randomUUID().toString

  def genSym(base: String): Sym = Sym.gen(base)

  // Build consistent expression for a filter-condition with keep polarity,
  // using Let to manage missing-ness.
  def filterPredicateWithKeep(irPred: IR, keep: Boolean): IR =
    bindIR(if (keep) irPred else ApplyUnaryPrimOp(Bang, irPred)) { pred =>
      If(IsNA(pred), False(), pred)
    }

  def invoke(name: String, rt: Type, typeArgs: Seq[Type], errorID: Int, args: IR*): IR =
    IRFunctionRegistry.lookupUnseeded(name, rt, typeArgs, args.map(_.typ)) match {
      case Some(f) => f(args, errorID)
      case None => fatal(
          s"no conversion found for $name[${typeArgs.mkString(", ")}](${args.map(_.typ).mkString(", ")}) => $rt"
        )
    }

  def invoke(name: String, rt: Type, typeArgs: Seq[Type], args: IR*): IR =
    invoke(name, rt, typeArgs, ErrorIDs.NO_ERROR, args: _*)

  def invoke(name: String, rt: Type, args: IR*): IR =
    invoke(name, rt, Array.empty[Type], ErrorIDs.NO_ERROR, args: _*)

  def invoke(name: String, rt: Type, errorID: Int, args: IR*): IR =
    invoke(name, rt, Array.empty[Type], errorID, args: _*)

  def invokeSeeded(name: String, staticUID: Long, rt: Type, rngState: IR, args: IR*): IR =
    IRFunctionRegistry.lookupSeeded(name, staticUID, rt, args.map(_.typ)) match {
      case Some(f) => f(args, rngState)
      case None =>
        fatal(s"no seeded function found for $name(${args.map(_.typ).mkString(", ")}) => $rt")
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
    val bindings = values.toFastSeq.map(freshName() -> _)
    Let(bindings, body(bindings.map(b => Ref(b._1, b._2.typ))))
  }

  def bindIR(v: IR)(body: Ref => IR): IR =
    bindIRs(v) { case Seq(ref) => body(ref) }

  def relationalBindIR(v: IR)(body: RelationalRef => IR): IR = {
    val ref = RelationalRef(freshName(), v.typ)
    RelationalLet(ref.name, v, body(ref))
  }

  def iota(start: IR, step: IR): IR = StreamIota(start, step)

  def dropWhile(v: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TStream](v.typ).elementType)
    StreamDropWhile(v, ref.name, f(ref))
  }

  def takeWhile(v: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TStream](v.typ).elementType)
    StreamTakeWhile(v, ref.name, f(ref))
  }

  def maxIR(a: IR, b: IR): IR =
    If(a > b, a, b)

  def minIR(a: IR, b: IR): IR =
    If(a < b, a, b)

  def streamAggIR(stream: IR)(f: Ref => IR): StreamAgg = {
    val ref = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    StreamAgg(stream, ref.name, f(ref))
  }

  def streamAggScanIR(stream: IR)(f: Ref => IR): StreamAggScan = {
    val ref = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    StreamAggScan(stream, ref.name, f(ref))
  }

  def forIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    StreamFor(stream, ref.name, f(ref))
  }

  def filterIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    StreamFilter(stream, ref.name, f(ref))
  }

  def mapIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    StreamMap(stream, ref.name, f(ref))
  }

  def mapArray(array: IR)(f: Ref => IR): IR =
    ToArray(mapIR(ToStream(array))(f))

  def flatMapIR(stream: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    StreamFlatMap(stream, ref.name, f(ref))
  }

  def flatten(stream: IR): IR =
    flatMapIR(if (stream.typ.isInstanceOf[TStream]) stream else ToStream(stream)) { elt =>
      if (elt.typ.isInstanceOf[TStream]) elt else ToStream(elt)
    }

  def foldIR(stream: IR, zero: IR)(f: (Ref, Ref) => IR): StreamFold = {
    val elt = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    val accum = Ref(freshName(), zero.typ)
    StreamFold(stream, zero, accum.name, elt.name, f(accum, elt))
  }

  def fold2IR(
    stream: IR,
    inits: IR*
  )(
    seqs: ((Ref, IndexedSeq[Ref]) => IR)*
  )(
    result: IndexedSeq[Ref] => IR
  ): IR = {
    val elt = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    val accums = inits.map(i => Ref(freshName(), i.typ)).toFastSeq
    StreamFold2(
      stream,
      accums.lazyZip(inits).map((acc, i) => (acc.name, i)),
      elt.name,
      seqs.map(f => f(elt, accums)).toFastSeq,
      result(accums),
    )
  }

  def streamScanIR(stream: IR, zero: IR)(f: (Ref, Ref) => IR): IR = {
    val elt = Ref(freshName(), tcoerce[TStream](stream.typ).elementType)
    val accum = Ref(freshName(), zero.typ)
    StreamScan(stream, zero, accum.name, elt.name, f(accum, elt))
  }

  def sortIR(stream: IR)(f: (Ref, Ref) => IR): IR = {
    val t = tcoerce[TStream](stream.typ).elementType
    val l = Ref(freshName(), t)
    val r = Ref(freshName(), t)
    ArraySort(stream, l.name, r.name, f(l, r))
  }

  def sliceArrayIR(arrayIR: IR, startIR: IR, stopIR: IR): IR =
    ArraySlice(arrayIR, startIR, Some(stopIR))

  def joinIR(
    left: IR,
    right: IR,
    lkey: IndexedSeq[String],
    rkey: IndexedSeq[String],
    joinType: String,
    requiresMemoryManagement: Boolean = false,
    rightKeyIsDistinct: Boolean = true,
  )(
    f: (Ref, Ref) => IR
  ): IR = {
    val lRef = Ref(freshName(), left.typ.asInstanceOf[TStream].elementType)
    val rRef = Ref(freshName(), right.typ.asInstanceOf[TStream].elementType)
    StreamJoin(
      left,
      right,
      lkey,
      rkey,
      lRef.name,
      rRef.name,
      f(lRef, rRef),
      joinType,
      requiresMemoryManagement,
      rightKeyIsDistinct,
    )
  }

  def zipJoin2IR(streams: IndexedSeq[IR], key: IndexedSeq[String])(f: (Ref, Ref) => IR): IR = {
    val eltType = tcoerce[TStruct](elementType(streams.head.typ))
    val curKey = Ref(freshName(), eltType.typeAfterSelectNames(key))
    val curVals = Ref(freshName(), TArray(eltType))
    StreamZipJoin(streams, key, curKey.name, curVals.name, f(curKey, curVals))
  }

  def joinRightDistinctIR(
    left: IR,
    right: IR,
    lkey: IndexedSeq[String],
    rkey: IndexedSeq[String],
    joinType: String,
  )(
    f: (Ref, Ref) => IR
  ): IR = {
    val lRef = Ref(freshName(), left.typ.asInstanceOf[TStream].elementType)
    val rRef = Ref(freshName(), right.typ.asInstanceOf[TStream].elementType)
    StreamJoinRightDistinct(left, right, lkey, rkey, lRef.name, rRef.name, f(lRef, rRef), joinType)
  }

  def streamSumIR(stream: IR): IR =
    foldIR(stream, 0) { case (accum, elt) => accum + elt }

  def streamForceCount(stream: IR): IR =
    streamSumIR(mapIR(stream)(_ => I32(1)))

  def rangeIR(n: IR): IR = StreamRange(0, n, 1)

  def rangeIR(start: IR, stop: IR): IR = StreamRange(start, stop, 1)

  def insertIR(old: IR, fields: (String, IR)*): InsertFields =
    InsertFields(old, fields.toArray[(String, IR)])

  def selectIR(old: IR, fields: String*): SelectFields = SelectFields(old, fields.toArray[String])

  def zip2(s1: IR, s2: IR, behavior: ArrayZipBehavior.ArrayZipBehavior)(f: (Ref, Ref) => IR): IR = {
    val r1 = Ref(freshName(), tcoerce[TStream](s1.typ).elementType)
    val r2 = Ref(freshName(), tcoerce[TStream](s2.typ).elementType)
    StreamZip(FastSeq(s1, s2), FastSeq(r1.name, r2.name), f(r1, r2), behavior)
  }

  def zipWithIndex(s: IR): IR = {
    val r1 = Ref(freshName(), tcoerce[TStream](s.typ).elementType)
    val r2 = Ref(freshName(), TInt32)
    StreamZip(
      FastSeq(s, StreamIota(I32(0), I32(1))),
      FastSeq(r1.name, r2.name),
      MakeStruct(FastSeq(("elt", r1), ("idx", r2))),
      ArrayZipBehavior.TakeMinLength,
    )
  }

  def zipIR(
    ss: IndexedSeq[IR],
    behavior: ArrayZipBehavior.ArrayZipBehavior,
    errorId: Int = ErrorIDs.NO_ERROR,
  )(
    f: IndexedSeq[Ref] => IR
  ): IR = {
    val refs = ss.map(s => Ref(freshName(), tcoerce[TStream](s.typ).elementType))
    StreamZip(ss, refs.map(_.name), f(refs), behavior, errorId)
  }

  def ndMap(nd: IR)(f: Ref => IR): IR = {
    val ref = Ref(freshName(), tcoerce[TNDArray](nd.typ).elementType)
    NDArrayMap(nd, ref.name, f(ref))
  }

  def ndMap2(nd1: IR, nd2: IR)(f: (Ref, Ref) => IR): IR = {
    val ref1 = Ref(freshName(), tcoerce[TNDArray](nd1.typ).elementType)
    val ref2 = Ref(freshName(), tcoerce[TNDArray](nd2.typ).elementType)
    NDArrayMap2(nd1, nd2, ref1.name, ref2.name, f(ref1, ref2), ErrorIDs.NO_ERROR)
  }

  def bmMap(bm: BlockMatrixIR, needsDense: Boolean)(f: Ref => IR): BlockMatrixMap = {
    val ref = Ref(freshName(), bm.typ.elementType)
    BlockMatrixMap(bm, ref.name, f(ref), needsDense)
  }

  def makestruct(fields: (String, IR)*): MakeStruct = MakeStruct(fields.toArray[(String, IR)])

  def maketuple(fields: IR*): MakeTuple =
    MakeTuple(fields.toArray.zipWithIndex.map { case (field, idx) => (idx, field) })

  def aggBindIR(v: IR, isScan: Boolean = false)(body: Ref => IR): IR = {
    val ref = Ref(freshName(), v.typ)
    AggLet(ref.name, v, body(ref), isScan = isScan)
  }

  def aggExplodeIR(v: IR, isScan: Boolean = false)(body: Ref => IR): AggExplode = {
    val r = Ref(freshName(), v.typ.asInstanceOf[TIterable].elementType)
    AggExplode(v, r.name, body(r), isScan)
  }

  def aggArrayPerElement(
    v: IR,
    knownLength: Option[IR] = None,
    isScan: Boolean = false,
  )(
    body: (Ref, Ref) => IR
  ): AggArrayPerElement = {
    val elt = Ref(freshName(), v.typ.asInstanceOf[TIterable].elementType)
    val idx = Ref(freshName(), TInt32)
    AggArrayPerElement(v, elt.name, idx.name, body(elt, idx), knownLength, isScan)
  }

  def aggFoldIR(zero: IR, isScan: Boolean = false)(seqOp: Ref => IR)(combOp: (Ref, Ref) => IR)
    : AggFold = {
    val accum1 = Ref(freshName(), zero.typ)
    val accum2 = Ref(freshName(), zero.typ)
    AggFold(zero, seqOp(accum1), combOp(accum1, accum2), accum1.name, accum2.name, isScan)
  }

  def cdaIR(
    contexts: IR,
    globals: IR,
    staticID: String,
    dynamicID: IR = NA(TString),
    tsd: Option[TableStageDependency] = None,
  )(
    body: (Ref, Ref) => IR
  ): CollectDistributedArray = {
    val contextRef = Ref(freshName(), contexts.typ.asInstanceOf[TStream].elementType)
    val globalRef = Ref(freshName(), globals.typ)

    CollectDistributedArray(
      contexts,
      globals,
      contextRef.name,
      globalRef.name,
      body(contextRef, globalRef),
      dynamicID,
      staticID,
      tsd,
    )
  }

  def tailLoop(resultType: Type, inits: IR*)(f: (IndexedSeq[IR] => IR, IndexedSeq[Ref]) => IR)
    : IR = {
    val loopName = freshName()
    val vars = inits.toFastSeq.map(x => Ref(freshName(), x.typ))
    def recur(vs: IndexedSeq[IR]): IR = Recur(loopName, vs, resultType)
    TailLoop(loopName, vars.map(_.name).zip(inits), resultType, f(recur, vars))
  }

  def mapPartitions(child: TableIR)(f: (Ref, Ref) => IR): TableMapPartitions = {
    val globals = Ref(freshName(), child.typ.globalType)
    val part = Ref(freshName(), TStream(child.typ.rowType))
    TableMapPartitions(child, globals.name, part.name, f(globals, part))
  }

  def mapPartitions(child: TableIR, requestedKey: Int, allowedOverlap: Int)(f: (Ref, Ref) => IR)
    : TableMapPartitions = {
    val globals = Ref(freshName(), child.typ.globalType)
    val part = Ref(freshName(), TStream(child.typ.rowType))
    TableMapPartitions(
      child,
      globals.name,
      part.name,
      f(globals, part),
      requestedKey,
      allowedOverlap,
    )
  }

  def tableGen(
    contexts: IR,
    globals: IR,
    partitioner: RVDPartitioner,
    errorID: Int = ErrorIDs.NO_ERROR,
  )(
    f: (Ref, Ref) => IR
  ): TableGen = {
    TypeCheck.coerce[TStream]("contexts", contexts.typ): Unit
    val c = Ref(freshName(), elementType(contexts.typ))
    val g = Ref(freshName(), globals.typ)
    TableGen(contexts, globals, c.name, g.name, f(c, g), partitioner, errorID)
  }

  def strConcat(irs: AnyRef*): IR = {
    assert(irs.nonEmpty)
    var s: IR = null
    irs.foreach { xAny =>
      val x = xAny match {
        case x: IR => x
        case x: String => Str(x)
      }

      val xstr = if (x.typ == TString)
        x
      else
        invoke("str", TString, x)

      if (s == null)
        s = xstr
      else
        s = invoke("concat", TString, s, xstr)
    }
    s
  }

  def logIR(result: IR, messages: AnyRef*): IR = ConsoleLog(strConcat(messages: _*), result)

  implicit def toRichIndexedSeqEmitSettable(s: IndexedSeq[EmitSettable])
    : RichIndexedSeqEmitSettable = new RichIndexedSeqEmitSettable(s)

  implicit def emitValueToCode(ev: EmitValue): EmitCode = ev.load

  implicit def toCodeParamType(ti: TypeInfo[_]): CodeParamType = CodeParamType(ti)

  implicit def toCodeParam(c: Value[_]): CodeParam = CodeParam(c)

  implicit def sValueToSCodeParam(sv: SValue): SCodeParam = SCodeParam(sv)

  implicit def toEmitParam(ec: EmitCode): EmitParam = EmitParam(ec)

  implicit def emitValueToEmitParam(ev: EmitValue): EmitParam = EmitParam(ev)
}
