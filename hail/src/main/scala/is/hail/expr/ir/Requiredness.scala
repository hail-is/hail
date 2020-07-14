package is.hail.expr.ir

import is.hail.expr.ir.functions.GetElement
import is.hail.methods.ForceCountTable
import is.hail.types._
import is.hail.types.physical.{PStream, PType}
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable

object Requiredness {
  def apply(node: BaseIR, usesAndDefs: UsesAndDefs, ctx: ExecuteContext, env: Env[PType]): RequirednessAnalysis = {
    val pass = new Requiredness(usesAndDefs, ctx)
    pass.initialize(node, env)
    pass.run()
    pass.result()
  }

  def apply(node: BaseIR, ctx: ExecuteContext): RequirednessAnalysis =
    apply(node, ComputeUsesAndDefs(node), ctx, Env.empty)
}

case class RequirednessAnalysis(r: Memo[BaseTypeWithRequiredness], states: Memo[IndexedSeq[TypeWithRequiredness]]) {
  def lookup(node: BaseIR): BaseTypeWithRequiredness = r.lookup(node)
  def apply(node: IR): TypeWithRequiredness = coerce[TypeWithRequiredness](lookup(node))
  def getState(node: IR): IndexedSeq[TypeWithRequiredness] = states(node)
}

class Requiredness(val usesAndDefs: UsesAndDefs, ctx: ExecuteContext) {
  type State = Memo[BaseTypeWithRequiredness]
  private val cache = Memo.empty[BaseTypeWithRequiredness]
  private val dependents = Memo.empty[mutable.Set[RefEquality[BaseIR]]]
  private val q = mutable.Set[RefEquality[BaseIR]]()

  private val defs = Memo.empty[IndexedSeq[BaseTypeWithRequiredness]]
  private val states = Memo.empty[IndexedSeq[TypeWithRequiredness]]

  def result(): RequirednessAnalysis = RequirednessAnalysis(cache, states)

  def lookup(node: IR): TypeWithRequiredness = coerce[TypeWithRequiredness](cache(node))
  def lookupAs[T <: TypeWithRequiredness](node: IR): T = coerce[T](cache(node))
  def lookup(node: TableIR): RTable = coerce[RTable](cache(node))

  def supportedType(node: BaseIR): Boolean = node.isInstanceOf[TableIR] || node.isInstanceOf[IR]

  private def initializeState(node: BaseIR): Unit = if (!cache.contains(node)) {
    assert(supportedType(node))
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
    node.children.foreach {
      case c: BlockMatrixIR => //ignore block matrices
      case c: MatrixIR => fatal("Requiredness analysis only works on lowered MatrixTables. ")
      case c if supportedType(node) =>
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

  def initialize(node: BaseIR, env: Env[PType]): Unit = {
    initializeState(node)
    usesAndDefs.uses.m.keys.foreach { n =>
      if (supportedType(n.t)) addBindingRelations(n.t)
    }
    if (usesAndDefs.free != null)
      usesAndDefs.free.foreach { re =>
        lookup(re.t).fromPType(env.lookup(re.t.name))
      }
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
    val refMap: Map[String, IndexedSeq[RefEquality[BaseRef]]] =
      usesAndDefs.uses(node).toFastIndexedSeq.groupBy(_.t.name)
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

    def addTableBinding(table: TableIR): Unit = {
      if (refMap.contains("row"))
        refMap("row").foreach { u => defs.bind(u, Array[BaseTypeWithRequiredness](lookup(table).rowType)) }
      if (refMap.contains("global"))
        refMap("global").foreach { u => defs.bind(u, Array[BaseTypeWithRequiredness](lookup(table).globalType)) }
      val refs = refMap.getOrElse("row", FastIndexedSeq()) ++ refMap.getOrElse("global", FastIndexedSeq())
      dependents.getOrElseUpdate(table, mutable.Set[RefEquality[BaseIR]]()) ++= refs
    }
    node match {
      case AggLet(name, value, body, isScan) => addBinding(name, value)
      case Let(name, value, body) => addBinding(name, value)
      case RelationalLet(name, value, body) => addBinding(name, value)
      case RelationalLetTable(name, value, body) => addBinding(name, value)
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
      case StreamZipJoin(as, key, curKey, curVals, _) =>
        val aEltTypes = as.map(a => coerce[RStruct](coerce[RIterable](lookup(a)).elementType))
        if (refMap.contains(curKey)) {
          val uses = refMap(curKey)
          val keyTypes = aEltTypes.map(t => RStruct(key.map(k => k -> t.fieldType(k))))
          uses.foreach { u => defs.bind(u, keyTypes) }
          as.foreach { a => dependents.getOrElseUpdate(a, mutable.Set[RefEquality[BaseIR]]()) ++= uses }
        }
        if (refMap.contains(curVals)) {
          val uses = refMap(curVals)
          val valTypes = aEltTypes.map { t =>
            val optional = t.copy(t.children)
            optional.union(false)
            RIterable(optional)
          }
          uses.foreach { u => defs.bind(u, valTypes) }
          as.foreach { a => dependents.getOrElseUpdate(a, mutable.Set[RefEquality[BaseIR]]()) ++= uses }
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
        if (refMap.contains(idx))
          refMap(idx).foreach { use => defs.bind(use, Array[BaseTypeWithRequiredness](RPrimitive())) }
      case NDArrayMap(nd, name, body) =>
        addElementBinding(name, nd)
      case NDArrayMap2(left, right, l, r, body) =>
        addElementBinding(l, left)
        addElementBinding(r, right)
      case CollectDistributedArray(ctxs, globs, c, g, body) =>
        addElementBinding(c, ctxs)
        addBinding(g, globs)

      case TableAggregate(c, q) =>
        addTableBinding(c)
      case TableFilter(child, pred) =>
        addTableBinding(child)
      case TableMapRows(child, newRow) =>
        addTableBinding(child)
      case TableMapGlobals(child, newGlobals) =>
        addTableBinding(child)
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        addTableBinding(child)
      case TableAggregateByKey(child, expr) =>
        addTableBinding(child)
      case _ => fatal(Pretty(node))
    }
  }

  def analyze(node: BaseIR): Boolean = node match {
    case x: IR => analyzeIR(x)
    case x: TableIR => analyzeTable(x)
    case _ =>
      fatal("MatrixTable and BlockMatrix must be lowered first.")
  }

  def analyzeTable(node: TableIR): Boolean = {
    val requiredness = lookup(node)
    node match {
      //statically known
      case TableLiteral(typ, rvd, enc, encodedGlobals) =>
        requiredness.rowType.fromPType(rvd.rowPType)
        requiredness.globalType.fromPType(enc.encodedType.decodedPType(typ.globalType))
      case TableRead(typ, dropRows, tr) =>
        val (rowPType, globalPType) = tr.rowAndGlobalPTypes(ctx, typ)
        requiredness.rowType.fromPType(rowPType)
        requiredness.globalType.fromPType(globalPType)
      case TableRange(_, _) =>

      // pass through TableIR child
      case TableKeyBy(child, _, _) => requiredness.unionFrom(lookup(child))
      case TableFilter(child, _) => requiredness.unionFrom(lookup(child))
      case TableHead(child, _) => requiredness.unionFrom(lookup(child))
      case TableTail(child, _) => requiredness.unionFrom(lookup(child))
      case TableRepartition(child, n, strategy) => requiredness.unionFrom(lookup(child))
      case TableDistinct(child) => requiredness.unionFrom(lookup(child))
      case TableOrderBy(child, sortFields) => requiredness.unionFrom(lookup(child))
      case TableRename(child, rMap, gMap) => requiredness.unionFrom(lookup(child))
      case TableFilterIntervals(child, intervals, keep) => requiredness.unionFrom(lookup(child))
      case RelationalLetTable(name, value, body) => requiredness.unionFrom(lookup(body))

      case TableParallelize(rowsAndGlobal, _) =>
        val Seq(rowsReq: RIterable, globalReq: RStruct) = lookupAs[RBaseStruct](rowsAndGlobal).children
        requiredness.unionRows(coerce[RStruct](rowsReq.elementType))
        requiredness.unionGlobals(globalReq)
      case TableMapRows(child, newRow) =>
        requiredness.unionRows(lookupAs[RStruct](newRow))
        requiredness.unionGlobals(lookup(child))
      case TableMapGlobals(child, newGlobals) =>
        requiredness.unionRows(lookup(child))
        requiredness.unionGlobals(lookupAs[RStruct](newGlobals))
      case TableExplode(child, path) =>
        val childReq = lookup(child)
        requiredness.unionGlobals(childReq)
        var i = 0
        var newFields: TypeWithRequiredness = requiredness.rowType
        var childFields: TypeWithRequiredness = childReq.rowType
        while (i < path.length) {
          val explode = path(i)
          coerce[RStruct](newFields).fields.filter(f => f.name != explode)
            .foreach(f => f.typ.unionFrom(coerce[RStruct](childFields).field(f.name)))
          newFields = coerce[RStruct](newFields).field(explode)
          childFields = coerce[RStruct](childFields).field(explode)
          i += 1
        }
        newFields.unionFrom(coerce[RIterable](childFields).elementType)
      case TableUnion(children) =>
        requiredness.unionFrom(lookup(children.head))
        children.tail.foreach(c => requiredness.unionRows(lookup(c)))
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        requiredness.unionKeys(lookupAs[RStruct](newKey))
        requiredness.unionValues(lookupAs[RStruct](expr))
        requiredness.unionGlobals(lookup(child))
      case TableAggregateByKey(child, expr) =>
        requiredness.unionKeys(lookup(child))
        requiredness.unionValues(lookupAs[RStruct](expr))
        requiredness.unionGlobals(lookup(child))
      case TableJoin(left, right, joinType, joinKey) =>
        val leftReq = lookup(left).changeKey(left.typ.key.take(joinKey))
        val rightReq = lookup(right).changeKey(right.typ.key.take(joinKey))

        requiredness.unionValues(leftReq)
        requiredness.unionValues(rightReq)

        if (joinType == "outer" || joinType == "zip" || joinType == "left") {
          requiredness.key.zip(leftReq.key).foreach { case (k, rk) =>
            requiredness.field(k).unionFrom(leftReq.field(rk))
          }
          rightReq.valueFields.foreach(n => requiredness.field(n).union(r = false))
        }

        if (joinType == "outer" || joinType == "zip" || joinType == "right") {
          requiredness.key.zip(rightReq.key).foreach { case (k, rk) =>
            requiredness.field(k).unionFrom(rightReq.field(rk))
          }
          leftReq.valueFields.foreach(n => requiredness.field(n).union(r = false))
        }

        if (joinType == "inner")
          requiredness.key.take(joinKey).zipWithIndex.foreach { case (k, i) =>
            requiredness.field(k).unionWithIntersection(FastSeq(
              leftReq.field(leftReq.key(i)),
              rightReq.field(rightReq.key(i))))
          }

        requiredness.unionGlobals(leftReq.globalType)
        requiredness.unionGlobals(rightReq.globalType)

      case TableIntervalJoin(left, right, root, product) =>
        val lReq = lookup(left)
        val rReq = lookup(right)
        requiredness.unionKeys(lReq)
        requiredness.valueFields.filter(_ != root)
          .foreach(n => requiredness.field(n).unionFrom(lReq.field(n)))
        val joinField = if (product)
          requiredness.field(root).asInstanceOf[RIterable]
            .elementType.asInstanceOf[RStruct]
        else requiredness.field(root).asInstanceOf[RStruct]
        rReq.valueFields.foreach { n => joinField.field(n).unionFrom(rReq.field(n)) }
        requiredness.field(root).union(false)
        requiredness.unionGlobals(lReq)
      case TableMultiWayZipJoin(children, valueName, globalName) =>
        val valueStruct = coerce[RStruct](coerce[RIterable](requiredness.field(valueName)).elementType)
        val globalStruct = coerce[RStruct](coerce[RIterable](requiredness.field(globalName)).elementType)
        children.foreach { c =>
          val cReq = lookup(c)
          requiredness.unionKeys(cReq)
          cReq.valueFields.foreach(n => valueStruct.field(n).unionFrom(cReq.field(n)))
          globalStruct.unionFrom(cReq.globalType)
        }
        valueStruct.union(false)
      case TableLeftJoinRightDistinct(left, right, root) =>
        val lReq = lookup(left)
        val rReq = lookup(right)
        requiredness.unionRows(lReq)
        val joined = coerce[RStruct](requiredness.field(root))
        rReq.valueFields.foreach(n => joined.field(n).unionFrom(rReq.field(n)))
        joined.union(false)
        requiredness.unionGlobals(lReq.globalType)
      case TableGroupWithinPartitions(child, name, n) =>
        val cReq = lookup(child)
        requiredness.unionKeys(cReq)
        val valueStruct = coerce[RStruct](coerce[RIterable](requiredness.field(name)).elementType)
        cReq.valueFields.foreach(n => valueStruct.field(n).unionFrom(cReq.field(n)))
        requiredness.unionGlobals(cReq.globalType)
      case TableToTableApply(child, function) => requiredness.maximize() //FIXME: needs implementation
      case BlockMatrixToTableApply(child, _, function) => requiredness.maximize() //FIXME: needs implementation
      case BlockMatrixToTable(child) => //all required
    }
    requiredness.probeChangedAndReset()
  }

  def analyzeIR(node: IR): Boolean = {
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
      case _: I32 | _: I64 | _: F32 | _: F64 | _: Str | True() | False() | _: IsNA | _: Die | _: UUID4 =>
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
      case StreamMerge(l, r, key) =>
        val lType = lookupAs[RIterable](l)
        val rType = lookupAs[RIterable](r)
        requiredness.union(lType.required && rType.required)
        coerce[RIterable](requiredness)
          .elementType
          .unionFrom(FastSeq(lType.elementType, rType.elementType))
      case StreamZip(as, names, body, behavior) =>
        requiredness.union(as.forall(lookup(_).required))
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(body))
      case StreamZipJoin(as, _, curKey, curVals, joinF) =>
        requiredness.union(as.forall(lookup(_).required))
        val eltType = coerce[RIterable](requiredness).elementType
        eltType.unionFrom(lookup(joinF))
      case StreamMultiMerge(as, _) =>
        requiredness.union(as.forall(lookup(_).required))
        coerce[RIterable](requiredness).elementType.unionFrom(as.map(a => coerce[RIterable](lookup(a)).elementType))
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
        val pResult = agg.PhysicalAggSig(aggSig.op, agg.AggStateSig(aggSig.op,
          initOpArgs.map(i => i -> lookup(i)),
          seqOpArgs.map(s => s -> lookup(s)))).pResultType
        requiredness.fromPType(pResult)
      case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
        val pResult = agg.PhysicalAggSig(aggSig.op, agg.AggStateSig(aggSig.op,
          initOpArgs.map(i => i -> lookup(i)),
          seqOpArgs.map(s => s -> lookup(s)))).pResultType
        requiredness.fromPType(pResult)
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
      case WritePartition(value, writeCtx, writer) =>
        val sType = coerce[PStream](lookup(value).canonicalPType(value.typ))
        val ctxType = lookup(writeCtx).canonicalPType(writeCtx.typ)
        requiredness.fromPType(writer.returnPType(ctxType, sType))
      case ReadValue(path, spec, rt) =>
        requiredness.union(lookup(path).required)
        requiredness.fromPType(spec.encodedType.decodedPType(rt))
      case In(_, t) => requiredness.fromPType(t)
      case LiftMeOut(f) => requiredness.unionFrom(lookup(f))
      case ResultOp(_, sigs) =>
        val r = coerce[RBaseStruct](requiredness)
        r.fields.foreach { f => f.typ.fromPType(sigs(f.index).pResultType) }
      case RunAgg(_, result, _) =>
        requiredness.unionFrom(lookup(result))
      case RunAggScan(array, name, init, seqs, result, signature) =>
        requiredness.union(lookup(array).required)
        coerce[RIterable](requiredness).elementType.unionFrom(lookup(result))
      case TableAggregate(c, q) => requiredness.unionFrom(lookup(q))
      case TableGetGlobals(c) => requiredness.unionFrom(lookup(c).globalType)
      case TableCollect(c) =>
        val cReq = lookup(c)
        val row = requiredness.asInstanceOf[RStruct].fieldType("rows").asInstanceOf[RIterable].elementType
        val global = requiredness.asInstanceOf[RStruct].fieldType("global")
        row.unionFrom(cReq.rowType)
        global.unionFrom(cReq.globalType)
      case TableToValueApply(c, f) =>
        f.unionRequiredness(lookup(c), requiredness)
      case TableGetGlobals(c) =>
        requiredness.unionFrom(lookup(c).globalType)
      case TableCollect(c) =>
        coerce[RIterable](coerce[RStruct](requiredness).field("rows")).elementType.unionFrom(lookup(c).rowType)
        coerce[RStruct](requiredness).field("global").unionFrom(lookup(c).globalType)
      case BlockMatrixToValueApply(child, GetElement(_)) => // BlockMatrix elements are all required
      case BlockMatrixCollect(child) =>  // BlockMatrix elements are all required
    }
    requiredness.probeChangedAndReset()
  }
}
