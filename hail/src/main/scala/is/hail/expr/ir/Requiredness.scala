package is.hail.expr.ir

import is.hail.methods.ForceCountTable
import is.hail.types._
import is.hail.types.physical.PType
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable

object Requiredness {
  def apply(node: BaseIR, usesAndDefs: UsesAndDefs, ctx: ExecuteContext, env: Env[PType], aggState: Option[Array[AggStatePhysicalSignature]]): RequirednessAnalysis = {
    val pass = new Requiredness(usesAndDefs, ctx)
    pass.initialize(node, env, aggState)
    pass.run()
    pass.result()
  }

  def apply(node: BaseIR, ctx: ExecuteContext): RequirednessAnalysis =
    apply(node, ComputeUsesAndDefs(node), ctx, Env.empty, None)
}

case class RequirednessAnalysis(r: Memo[BaseTypeWithRequiredness], states: Memo[Array[TypeWithRequiredness]]) {
  def lookup(node: BaseIR): BaseTypeWithRequiredness = r.lookup(node)
  def apply(node: IR): TypeWithRequiredness = coerce[TypeWithRequiredness](lookup(node))
  def getState(node: IR): Array[TypeWithRequiredness] = states(node)
}

class Requiredness(val usesAndDefs: UsesAndDefs, ctx: ExecuteContext) {
  type State = Memo[BaseTypeWithRequiredness]
  private val cache = Memo.empty[BaseTypeWithRequiredness]
  private val dependents = Memo.empty[mutable.Set[RefEquality[BaseIR]]]
  private val q = mutable.Set[RefEquality[BaseIR]]()

  private val defs = Memo.empty[Array[BaseTypeWithRequiredness]]
  private val states = Memo.empty[Array[TypeWithRequiredness]]

  private class WrappedAggState(val scope: RefEquality[IR], private var _sig: Array[AggStatePhysicalSignature]) {
    private var changed: Boolean = false
    def probeChangedAndReset(): Boolean = {
      val r = changed
      changed = false
      r
    }

    def matchesSignature(newSig: Array[AggStatePhysicalSignature]): Boolean =
      sig != null && (newSig sameElements sig)

    def setSignature(newSig: Array[AggStatePhysicalSignature]): Unit = {
      if (_sig == null || !(newSig sameElements _sig)) {
        changed = true
        _sig = newSig
      }
    }

    def sig: Array[AggStatePhysicalSignature] = _sig
  }

  private[this] val aggStateMemo = Memo.empty[WrappedAggState]

  private[this] def computeAggState(sigs: IndexedSeq[AggStateSignature], irs: Seq[IR]): Array[AggStatePhysicalSignature] = {
    val initsAB = InferPType.newBuilder[(AggOp, Seq[PType])](sigs.length)
    val seqsAB = InferPType.newBuilder[(AggOp, Seq[PType])](sigs.length)
    irs.foreach { ir => InferPType._extractAggOps(ir, inits = initsAB, seqs = seqsAB, Some(cache)) }
    Array.tabulate(sigs.length) { i => InferPType.computePhysicalAgg(sigs(i), initsAB(i), seqsAB(i)) }
  }

  private[this] def initializeAggStates(node: BaseIR, wrapped: WrappedAggState): Unit = {
    node match {
      case x@RunAgg(body, result, signature) =>
        val next = new WrappedAggState(RefEquality[IR](x), null)
        initializeAggStates(body, next)
        initializeAggStates(result, next)
        aggStateMemo.bind(x, next)
      case x@RunAggScan(array, name, init, seqs, result, signature) =>
        initializeAggStates(array, wrapped)
        val next = new WrappedAggState(RefEquality[IR](x), null)
        initializeAggStates(init, next)
        initializeAggStates(seqs, next)
        initializeAggStates(result, next)
        aggStateMemo.bind(x, next)
      case x@InitOp(_, args, _, _) =>
        aggStateMemo.bind(node, wrapped)
        q += RefEquality(x)
        args.foreach(a => dependents.getOrElseUpdate(a, mutable.Set[RefEquality[BaseIR]]()) += RefEquality(x))
        if (wrapped.scope != null)
          dependents.getOrElseUpdate(x, mutable.Set[RefEquality[BaseIR]]()) += wrapped.scope
        node.children.foreach(initializeAggStates(_, wrapped))
      case x@SeqOp(_, args, _, _) =>
        aggStateMemo.bind(node, wrapped)
        q += RefEquality(x)
        args.foreach(a => dependents.getOrElseUpdate(a, mutable.Set[RefEquality[BaseIR]]()) += RefEquality(x))
        if (wrapped.scope != null)
          dependents.getOrElseUpdate(x, mutable.Set[RefEquality[BaseIR]]()) += wrapped.scope
        node.children.foreach(initializeAggStates(_, wrapped))
      case x: ResultOp =>
        aggStateMemo.bind(node, wrapped)
        if (wrapped.scope != null)
          dependents.getOrElseUpdate(wrapped.scope, mutable.Set[RefEquality[BaseIR]]()) += RefEquality(x)
      case _ => node.children.foreach(initializeAggStates(_, wrapped))
    }
  }

  private[this] def lookupAggState(ir: IR): WrappedAggState = aggStateMemo(ir)

  def result(): RequirednessAnalysis = RequirednessAnalysis(cache, states)

  def lookup(node: IR): TypeWithRequiredness = coerce[TypeWithRequiredness](cache(node))
  def lookupAs[T <: TypeWithRequiredness](node: IR): T = coerce[T](cache(node))

  private def initializeState(node: BaseIR): Unit = if (!cache.contains(node)) {
    val re = RefEquality(node)
    node match {
      case x: ApplyIR =>
        initializeState(x.body)
        val xUses = ComputeUsesAndDefs(x.body, errorIfFreeVariables = false)
        xUses.uses.m.foreach { case (re, uses) =>
          usesAndDefs.uses.bind(re, uses)
        }
        usesAndDefs.uses.bind(re, xUses.free)
        dependents.getOrElseUpdate(x.body, mutable.Set[RefEquality[BaseIR]]()) += re
      case _ =>
    }
    node.children.foreach { c =>
      initializeState(c)
      if (node.typ != TVoid)
        dependents.getOrElseUpdate(c, mutable.Set[RefEquality[BaseIR]]()) += re
    }
    if (node.typ != TVoid) {
      cache.bind(node, BaseTypeWithRequiredness(node.typ))
      if (usesAndDefs.free == null || !re.t.isInstanceOf[BaseRef] || !usesAndDefs.free.contains(re.asInstanceOf[RefEquality[BaseRef]]))
        q += re
    }
  }

  def initialize(node: BaseIR, env: Env[PType], outerAggStates: Option[Array[AggStatePhysicalSignature]]): Unit = {
    initializeState(node)
    usesAndDefs.uses.m.keys.foreach(n => addBindingRelations(n.t))
    if (usesAndDefs.free != null)
      usesAndDefs.free.foreach { re =>
        lookup(re.t).fromPType(env.lookup(re.t.name))
      }
    initializeAggStates(node, outerAggStates.map { sigs => new WrappedAggState(null, sigs) }.orNull)
  }

  def run(): Unit = {
    while (q.nonEmpty) {
      val node = q.head
      q -= node
      if (analyze(node.t) && dependents.contains(node)) {
        q ++= dependents.lookup(node)
      }
    }
  }

  def addBindingRelations(node: BaseIR): Unit = {
    val refMap = usesAndDefs.uses(node).toArray.groupBy(_.t.name).mapValues(_.asInstanceOf[Array[RefEquality[BaseIR]]])
    def addElementBinding(name: String, d: IR, makeOptional: Boolean = false): Unit = {
      if (refMap.contains(name)) {
        val uses = refMap(name)
        val eltReq = coerce[RIterable](lookup(d)).elementType
        val req = if (makeOptional) {
          val optional = eltReq.copy(eltReq.children)
          optional.union(false)
          optional
        } else eltReq
        uses.foreach { u => defs.bind(u, Array(req)) }
        dependents.getOrElseUpdate(d, mutable.Set[RefEquality[BaseIR]]()) ++= uses
      }
    }

    def addBindings(name: String, ds: Array[IR]): Unit = {
      if (refMap.contains(name)) {
        val uses = refMap(name)
        uses.foreach { u => defs.bind(u, ds.map(lookup).toArray[BaseTypeWithRequiredness]) }
        ds.foreach { d => dependents.getOrElseUpdate(d, mutable.Set[RefEquality[BaseIR]]()) ++= uses }
      }
    }

    def addBinding(name: String, ds: IR): Unit =
      addBindings(name, Array(ds))

    node match {
      case Let(name, value, body) => addBinding(name, value)
      case TailLoop(loopName, params, body) =>
        addBinding(loopName, body)
        val argDefs = Array.fill(params.length)(new ArrayBuilder[IR]())
        refMap(loopName).map(_.t).foreach { case Recur(_, args, _) =>
          argDefs.zip(args).foreach { case (ab, d) => ab += d }
        }
        val s = Array.fill[TypeWithRequiredness](params.length)(null)
        var i = 0
        while (i < params.length) {
          val (name, init) = params(i)
          s(i) = lookup(refMap.get(name).flatMap(refs => refs.headOption.map(_.t.asInstanceOf[IR])).getOrElse(init))
          addBindings(name, argDefs(i).result() :+ init)
          i += 1
        }
        states.bind(node, s)
      case x@ApplyIR(_, _, args) =>
        x.refIdx.foreach { case (n, i) => addBinding(n, args(i)) }
      case ArraySort(a, l, r, c) =>
        addElementBinding(l, a)
        addElementBinding(r, a)
      case StreamMap(a, name, body) =>
        addElementBinding(name, a)
      case x@StreamZip(as, names, body, behavior) =>
        var i = 0
        while (i < names.length) {
          addElementBinding(names(i), as(i),
            makeOptional = behavior == ArrayZipBehavior.ExtendNA)
          i += 1
        }
      case StreamFilter(a, name, cond) => addElementBinding(name, a)
      case StreamFlatMap(a, name, body) => addElementBinding(name, a)
      case StreamFor(a, name, _) => addElementBinding(name, a)
      case StreamFold(a, zero, accumName, valueName, body) =>
        addElementBinding(valueName, a)
        addBindings(accumName, Array[IR](zero, body))
        states.bind(node, Array[TypeWithRequiredness](lookup(
          refMap.get(accumName)
            .flatMap(refs => refs.headOption.map(_.t.asInstanceOf[IR]))
            .getOrElse(zero))))
      case StreamScan(a, zero, accumName, valueName, body) =>
        addElementBinding(valueName, a)
        addBindings(accumName, Array[IR](zero, body))
        states.bind(node, Array[TypeWithRequiredness](lookup(
          refMap.get(accumName)
            .flatMap(refs => refs.headOption.map(_.t.asInstanceOf[IR]))
            .getOrElse(zero))))
      case StreamFold2(a, accums, valueName, seq, result) =>
        addElementBinding(valueName, a)
        val s = Array.fill[TypeWithRequiredness](accums.length)(null)
        var i = 0
        while (i < accums.length) {
          val (n, z) = accums(i)
          addBindings(n, Array[IR](z, seq(i)))
          s(i) = lookup(refMap.get(n).flatMap(refs => refs.headOption.map(_.t.asInstanceOf[IR])).getOrElse(z))
          i += 1
        }
        states.bind(node, s)
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, joinf, joinType) =>
        addElementBinding(l, left, makeOptional = (joinType == "outer"))
        addElementBinding(r, right, makeOptional = true)
      case StreamAgg(a, name, query) =>
        addElementBinding(name, a)
      case StreamAggScan(a, name, query) =>
        addElementBinding(name, a)
      case RunAggScan(a, name, init, seqs, result, signature) =>
        addElementBinding(name, a)
      case AggExplode(a, name, aggBody, isScan) =>
        addElementBinding(name, a)
      case AggArrayPerElement(a, elt, idx, body, knownLength, isScan) =>
        addElementBinding(elt, a)
      //idx is always required Int32
      case NDArrayMap(nd, name, body) =>
        addElementBinding(name, nd)
      case NDArrayMap2(left, right, l, r, body) =>
        addElementBinding(l, left)
        addElementBinding(r, right)
      case CollectDistributedArray(ctxs, globs, c, g, body) =>
        addElementBinding(c, ctxs)
        addBinding(g, globs)
    }
  }

  def analyze(node: BaseIR): Boolean = node match {
    case x: IR => analyzeIR(x)
    case x: TableIR => fatal("Table nodes not yet supported.")
    case x: BlockMatrixIR => fatal("BM nodes not yet supported.")
    case _ =>
      fatal("MatrixTable must be lowered first.")
  }

  def analyzeIR(node: IR): Boolean = {
    if (node.typ == TVoid)
      return (node.isInstanceOf[InitOp] || node.isInstanceOf[SeqOp]) && (lookupAggState(node).scope != null)
    val requiredness = lookup(node)
    node match {
      // union all
      case _: Cast |
           _: CastRename |
           _: ToSet |
           _: CastToArray |
           _: ToArray |
           _: ToStream |
           _: NDArrayReindex |
           _: NDArrayAgg =>
        node.children.foreach { case c: IR => requiredness.unionFrom(lookup(c)) }

      // union top-level
      case _: ApplyBinaryPrimOp |
           _: ApplyUnaryPrimOp |
           _: ArrayLen |
           _: StreamLen |
           _: ArrayZeros |
           _: StreamRange |
           _: WriteValue =>
        requiredness.union(node.children.forall { case c: IR => lookup(c).required })
      case x: ApplyComparisonOp if x.op.strict =>
        requiredness.union(node.children.forall { case c: IR => lookup(c).required })

      // always required
      case _: I32 | _: I64 | _: F32 | _: F64 | _: Str | True() | False() | _: IsNA | _: Die =>
      case _: CombOpValue | _: AggStateValue =>
      case x if x.typ == TVoid =>
      case ApplyComparisonOp(EQWithNA(_, _), _, _) | ApplyComparisonOp(NEQWithNA(_, _), _, _) | ApplyComparisonOp(Compare(_, _), _, _) =>
      case ApplyComparisonOp(op, l, r) =>
        fatal(s"non-strict comparison op $op must have explicit case")
      case TableCount(t) =>
      case TableToValueApply(t, ForceCountTable()) =>

      case _: NA => requiredness.union(false)
      case Literal(t, a) => requiredness.unionLiteral(a)

      case Coalesce(values) =>
        val reqs = values.map(lookup)
        requiredness.union(reqs.exists(_.required))
        reqs.foreach(r => requiredness.children.zip(r.children).foreach { case (r1, r2) => r1.unionFrom(r2) })
      case If(cond, cnsq, altr) =>
        requiredness.union(lookup(cond).required)
        requiredness.unionFrom(lookup(cnsq))
        requiredness.unionFrom(lookup(altr))

      case AggLet(name, value, body, isScan) =>
        requiredness.unionFrom(lookup(body))
      case Let(name, value, body) =>
        requiredness.unionFrom(lookup(body))
      case RelationalLet(name, value, body) =>
        requiredness.unionFrom(lookup(body))
      case TailLoop(name, params, body) =>
        requiredness.unionFrom(lookup(body))
      case x: BaseRef =>
        requiredness.unionFrom(defs(node).map(coerce[TypeWithRequiredness]))
      case MakeArray(args, _) =>
        coerce[RIterable](requiredness).elementType.unionFrom(args.map(lookup))
      case MakeStream(args, _) =>
        coerce[RIterable](requiredness).elementType.unionFrom(args.map(lookup))
      case ArrayRef(a, i, _) =>
        val aReq = lookupAs[RIterable](a)
        requiredness.unionFrom(aReq.elementType)
        requiredness.union(lookup(i).required && aReq.required)
      case ArraySort(a, l, r, c) =>
        requiredness.unionFrom(lookup(a))
      case ToDict(a) =>
        val aReq = lookupAs[RIterable](a)
        val Seq(keyType, valueType) = coerce[RBaseStruct](aReq.elementType).children
        coerce[RDict](requiredness).keyType.unionFrom(keyType)
        coerce[RDict](requiredness).valueType.unionFrom(valueType)
        requiredness.union(aReq.required)
      case LowerBoundOnOrderedCollection(collection, elem, _) =>
        requiredness.union(lookup(collection).required)
      case GroupByKey(c) =>
        val cReq = lookupAs[RIterable](c)
        val Seq(k, v) = coerce[RBaseStruct](cReq.elementType).children
        coerce[RDict](requiredness).keyType.unionFrom(k)
        coerce[RIterable](coerce[RDict](requiredness).valueType).elementType.unionFrom(v)
        requiredness.union(cReq.required)
      case StreamGrouped(a, size) =>
        val aReq = lookupAs[RIterable](a)
        coerce[RIterable](coerce[RIterable](requiredness).elementType).elementType
          .unionFrom(aReq.elementType)
        requiredness.union(aReq.required && lookup(size).required)
      case StreamGroupByKey(a, key) =>
        val aReq = lookupAs[RIterable](a)
        coerce[RIterable](coerce[RIterable](requiredness).elementType).elementType
          .unionFrom(aReq.elementType)
        requiredness.union(aReq.required)
      case StreamMap(a, name, body) =>
        requiredness.union(lookup(a).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(body))
      case StreamTake(a, n) =>
        requiredness.union(lookup(n).required)
        requiredness.unionFrom(lookup(a))
      case StreamDrop(a, n) =>
        requiredness.union(lookup(n).required)
        requiredness.unionFrom(lookup(a))
      case StreamZip(as, names, body, behavior) =>
        requiredness.union(as.forall(lookup(_).required))
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(body))
      case StreamFilter(a, name, cond) =>
        requiredness.unionFrom(lookup(a))
      case StreamFlatMap(a, name, body) =>
        requiredness.union(lookup(a).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookupAs[RIterable](body).elementType)
      case StreamFold(a, zero, accumName, valueName, body) =>
        requiredness.union(lookup(a).required)
        requiredness.unionFrom(lookup(body))
        requiredness.unionFrom(lookup(zero)) // if a is length 0
      case StreamScan(a, zero, accumName, valueName, body) =>
        requiredness.union(lookup(a).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(body))
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(zero))
      case StreamFold2(a, accums, valueName, seq, result) =>
        requiredness.union(lookup(a).required)
        requiredness.unionFrom(lookup(result))
      case StreamJoinRightDistinct(left, right, _, _, _, _, joinf, joinType) =>
        requiredness.union(lookup(left).required && lookup(right).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(joinf))
      case StreamAgg(a, name, query) =>
        requiredness.union(lookup(a).required)
        requiredness.unionFrom(lookup(query))
      case StreamAggScan(a, name, query) =>
        requiredness.union(lookup(a).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(query))
      case AggFilter(cond, aggIR, isScan) =>
        requiredness.unionFrom(lookup(aggIR))
      case AggExplode(array, name, aggBody, isScan) =>
        requiredness.unionFrom(lookup(aggBody))
      case AggGroupBy(key, aggIR, isScan) =>
        val rdict = coerce[RDict](requiredness)
        rdict.keyType.unionFrom(lookup(key))
        rdict.valueType.unionFrom(lookup(aggIR))
      case AggArrayPerElement(a, _, _, body, knownLength, isScan) =>
        val rit = coerce[RIterable](requiredness)
        rit.union(lookup(a).required)
        rit.elementType.unionFrom(lookup(body))
      case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) => //FIXME round-tripping through ptype
        val initPTypes = initOpArgs.map(a => lookup(a).canonicalPType(a.typ))
        val seqPTypes = seqOpArgs.map(a => lookup(a).canonicalPType(a.typ))
        requiredness.fromPType(aggSig.toPhysical(initPTypes, seqPTypes).returnType)
      case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
        val initPTypes = initOpArgs.map(a => lookup(a).canonicalPType(a.typ))
        val seqPTypes = seqOpArgs.map(a => lookup(a).canonicalPType(a.typ))
        requiredness.fromPType(aggSig.toPhysical(initPTypes, seqPTypes).returnType)

      case MakeNDArray(data, shape, rowMajor) =>
        requiredness.unionFrom(lookup(data))
        requiredness.union(lookup(shape).required)
      case NDArrayShape(nd) =>
        requiredness.union(lookup(nd).required)
      case NDArrayReshape(nd, shape) =>
        val sReq = lookupAs[RBaseStruct](shape)
        val ndReq = lookup(nd)
        requiredness.unionFrom(ndReq)
        requiredness.union(sReq.required && sReq.children.forall(_.required))
      case NDArrayConcat(nds, axis) =>
        val ndsReq = lookupAs[RIterable](nds)
        requiredness.unionFrom(ndsReq.elementType)
        requiredness.union(ndsReq.required)
      case NDArrayRef(nd, idxs) =>
        val ndReq = lookupAs[RNDArray](nd)
        requiredness.unionFrom(ndReq.elementType)
        requiredness.union(ndReq.required && idxs.forall(lookup(_).required))
      case NDArraySlice(nd, slices) =>
        val slicesReq = lookupAs[RTuple](slices)
        val allSlicesRequired = slicesReq.fields.map(_.typ).forall {
          case r: RTuple => r.required && r.fields.forall(_.typ.required)
          case r => r.required
        }
        requiredness.unionFrom(lookup(nd))
        requiredness.union(slicesReq.required && allSlicesRequired)
      case NDArrayFilter(nd, keep) =>
        requiredness.unionFrom(lookup(nd))
      case NDArrayMap(nd, name, body) =>
        requiredness.union(lookup(nd).required)
        coerce[RNDArray](requiredness).unionElement(lookup(body))
      case NDArrayMap2(l, r, _, _, body) =>
        requiredness.union(lookup(l).required && lookup(r).required)
        coerce[RNDArray](requiredness).unionElement(lookup(body))
      case NDArrayMatMul(l, r) =>
        requiredness.unionFrom(lookup(l))
        requiredness.union(lookup(r).required)
      case NDArrayQR(nd, mode) => requiredness.fromPType(NDArrayQR.pTypes(mode))
      case MakeStruct(fields) =>
        fields.foreach { case (n, f) =>
          coerce[RStruct](requiredness).field(n).unionFrom(lookup(f))
        }
      case MakeTuple(fields) =>
        fields.foreach { case (i, f) =>
          coerce[RTuple](requiredness).field(i).unionFrom(lookup(f))
        }
      case SelectFields(old, fields) =>
        val oldReq = lookupAs[RStruct](old)
        requiredness.union(oldReq.required)
        fields.foreach { n =>
          coerce[RStruct](requiredness).field(n).unionFrom(oldReq.field(n))
        }
      case InsertFields(old, fields, _) =>
        lookup(old) match {
          case oldReq: RStruct =>
            requiredness.union(oldReq.required)
            val fieldMap = fields.toMap.mapValues(lookup)
            coerce[RStruct](requiredness).fields.foreach { f =>
              f.typ.unionFrom(fieldMap.getOrElse(f.name, oldReq.field(f.name)))
            }
          case _ => fields.foreach { case (n, f) =>
            coerce[RStruct](requiredness).field(n).unionFrom(lookup(f))
          }
        }
      case GetField(o, name) =>
        val oldReq = lookupAs[RStruct](o)
        requiredness.union(oldReq.required)
        requiredness.unionFrom(oldReq.field(name))
      case GetTupleElement(o, idx) =>
        val oldReq = lookupAs[RTuple](o)
        requiredness.union(oldReq.required)
        requiredness.unionFrom(oldReq.field(idx))
      case x: ApplyIR => requiredness.unionFrom(lookup(x.body))
      case x: AbstractApplyNode[_] => //FIXME: round-tripping via PTypes.
        val argP = x.args.map(a => lookup(a).canonicalPType(a.typ))
        requiredness.fromPType(x.implementation.returnPType(x.returnType, argP))
      case CollectDistributedArray(ctxs, globs, _, _, body) =>
        requiredness.union(lookup(ctxs).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(body))
      case ReadPartition(context, rowType, reader) =>
        requiredness.union(lookup(context).required)
        coerce[RIterable](requiredness).elementType.fromPType(reader.rowPType(rowType))
      case ReadValue(path, spec, rt) =>
        requiredness.union(lookup(path).required)
        requiredness.fromPType(spec.encodedType.decodedPType(rt))
      case In(_, t) => requiredness.fromPType(t)
      case LiftMeOut(f) => requiredness.unionFrom(lookup(f))
      case x@ResultOp(startIdx, aggSigs) =>
        val wrappedSigs = lookupAggState(x)
        if (wrappedSigs.sig != null) {
          val pTypes = Array.tabulate(aggSigs.length) { i => wrappedSigs.sig(startIdx + i).resultType }
          coerce[RBaseStruct](requiredness).fields.zip(pTypes).foreach { case (f, pt) =>
            f.typ.fromPType(pt)
          }
        }
      case x@RunAgg(body, result, signature) =>
        val wrapped = lookupAggState(x)
        val newAggSig = computeAggState(signature, FastSeq(body))
        if (!wrapped.matchesSignature(newAggSig))
          wrapped.setSignature(newAggSig)
        else
          requiredness.unionFrom(lookup(result))
      case x@RunAggScan(array, name, init, seqs, result, signature) =>
        requiredness.union(lookup(array).required)
        val wrapped = lookupAggState(x)
        val newAggSig = computeAggState(signature, FastSeq(init, seqs))
        if (!wrapped.matchesSignature(newAggSig))
          wrapped.setSignature(newAggSig)
        else
          coerce[RIterable](requiredness).elementType.unionFrom(lookup(result))
    }
    val aggScopeChanged = (node.isInstanceOf[RunAgg] || node.isInstanceOf[RunAggScan]) && (lookupAggState(node).probeChangedAndReset())
    requiredness.probeChangedAndReset() | aggScopeChanged
  }
}
