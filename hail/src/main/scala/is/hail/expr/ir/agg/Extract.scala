package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionPool, RegionValue}
import is.hail.asm4s.{HailClassLoader, _}
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.backend.spark.SparkTaskContext
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.io.BufferSpec
import is.hail.types.{TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.types.physical.stypes.EmitType
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.mutable

import org.apache.spark.TaskContext

class UnsupportedExtraction(msg: String) extends Exception(msg)

object AggStateSig {
  def apply(op: AggOp, seqOpArgs: Seq[IR], r: RequirednessAnalysis): AggStateSig = {
    val seqs = seqOpArgs.map(s => s -> (if (s.typ == TVoid) null else r(s)))
    apply(op, seqs)
  }

  // FIXME: factor out requiredness inference part
  def apply(
    op: AggOp,
    seqOpArgs: Seq[(IR, TypeWithRequiredness)],
  ): AggStateSig = {
    val seqVTypes = seqOpArgs.map { case (a, r) => VirtualTypeWithReq(a.typ, r) }
    op match {
      case Sum() | Product() => TypedStateSig(seqVTypes.head.setRequired(true))
      case Min() | Max() => TypedStateSig(seqVTypes.head.setRequired(false))
      case Count() => TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true))
      case Take() => TakeStateSig(seqVTypes.head)
      case ReservoirSample() => ReservoirSampleStateSig(seqVTypes.head)
      case Densify() => DensifyStateSig(seqVTypes.head)
      case TakeBy(reverse) =>
        val Seq(vt, kt) = seqVTypes
        TakeByStateSig(vt, kt, reverse)
      case CallStats() => CallStatsStateSig()
      case PrevNonnull() => TypedStateSig(seqVTypes.head.setRequired(false))
      case CollectAsSet() => CollectAsSetStateSig(seqVTypes.head)
      case Collect() => CollectStateSig(seqVTypes.head)
      case LinearRegression() => LinearRegressionStateSig()
      case ApproxCDF() => ApproxCDFStateSig()
      case Downsample() =>
        val Seq(_, _, labelType) = seqVTypes
        DownsampleStateSig(labelType)
      case ImputeType() => ImputeTypeStateSig()
      case NDArraySum() =>
        NDArraySumStateSig(
          seqVTypes.head.setRequired(false)
        ) // set required to false to handle empty aggs
      case NDArrayMultiplyAdd() => NDArrayMultiplyAddStateSig(seqVTypes.head.setRequired(false))
      case _ => throw new UnsupportedExtraction(op.toString)
    }
  }

  def grouped(k: IR, aggs: Seq[AggStateSig], r: RequirednessAnalysis): GroupedStateSig =
    GroupedStateSig(VirtualTypeWithReq(k.typ, r(k)), aggs)

  def getState(sig: AggStateSig, cb: EmitClassBuilder[_]): AggregatorState = sig match {
    case TypedStateSig(vt) if vt.t.isPrimitive => new PrimitiveRVAState(Array(vt), cb)
    case TypedStateSig(vt) => new TypedRegionBackedAggState(vt, cb)
    case DownsampleStateSig(labelType) => new DownsampleState(cb, labelType)
    case TakeStateSig(vt) => new TakeRVAS(vt, cb)
    case ReservoirSampleStateSig(vt) => new ReservoirSampleRVAS(vt, cb)
    case DensifyStateSig(vt) => new DensifyState(vt, cb)
    case TakeByStateSig(vt, kt, so) => new TakeByRVAS(vt, kt, cb, so)
    case CollectStateSig(pt) => new CollectAggState(pt, cb)
    case CollectAsSetStateSig(pt) => new AppendOnlySetState(cb, pt)
    case CallStatsStateSig() => new CallStatsState(cb)
    case ApproxCDFStateSig() => new ApproxCDFState(cb)
    case ImputeTypeStateSig() => new ImputeTypeState(cb)
    case ArrayAggStateSig(nested) => new ArrayElementState(
        cb,
        StateTuple(nested.map(sig => AggStateSig.getState(sig, cb)).toArray),
      )
    case GroupedStateSig(kt, nested) =>
      new DictState(cb, kt, StateTuple(nested.map(sig => AggStateSig.getState(sig, cb)).toArray))
    case NDArraySumStateSig(nda) => new TypedRegionBackedAggState(nda, cb)
    case NDArrayMultiplyAddStateSig(nda) =>
      new TypedRegionBackedAggState(nda, cb)
    case FoldStateSig(resultEmitType, _, _, _) =>
      val vWithReq = resultEmitType.typeWithRequiredness
      new TypedRegionBackedAggState(vWithReq, cb)
    case LinearRegressionStateSig() => new LinearRegressionAggregatorState(cb)
  }
}

sealed abstract class AggStateSig(val t: Seq[VirtualTypeWithReq], val n: Option[Seq[AggStateSig]])
case class TypedStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)

case class DownsampleStateSig(labelType: VirtualTypeWithReq)
    extends AggStateSig(Array(labelType), None)

case class TakeStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)

case class TakeByStateSig(vt: VirtualTypeWithReq, kt: VirtualTypeWithReq, so: SortOrder)
    extends AggStateSig(Array(vt, kt), None)

case class ReservoirSampleStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class DensifyStateSig(vt: VirtualTypeWithReq) extends AggStateSig(Array(vt), None)
case class CollectStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class CollectAsSetStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class CallStatsStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)
case class ImputeTypeStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)

case class ArrayAggStateSig(nested: Seq[AggStateSig])
    extends AggStateSig(Array[VirtualTypeWithReq](), Some(nested))

case class GroupedStateSig(kt: VirtualTypeWithReq, nested: Seq[AggStateSig])
    extends AggStateSig(Array(kt), Some(nested))

case class ApproxCDFStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)
case class LinearRegressionStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)

case class NDArraySumStateSig(nda: VirtualTypeWithReq)
    extends AggStateSig(Array[VirtualTypeWithReq](nda), None) {
  require(!nda.r.required)
}

case class NDArrayMultiplyAddStateSig(nda: VirtualTypeWithReq)
    extends AggStateSig(Array[VirtualTypeWithReq](nda), None) {
  require(!nda.r.required)
}

case class FoldStateSig(
  resultEmitType: EmitType,
  accumName: String,
  otherAccumName: String,
  combOpIR: IR,
) extends AggStateSig(Array[VirtualTypeWithReq](resultEmitType.typeWithRequiredness), None)

object PhysicalAggSig {
  def apply(op: AggOp, state: AggStateSig): PhysicalAggSig = BasicPhysicalAggSig(op, state)

  def unapply(v: PhysicalAggSig): Option[(AggOp, AggStateSig)] =
    if (v.nestedOps.isEmpty) Some(v.op -> v.state) else None
}

// A pair of an agg state and an op. If the state is compound, also encodes ops for nested states.
class PhysicalAggSig(val op: AggOp, val state: AggStateSig, val nestedOps: Array[AggOp]) {
  val allOps: Array[AggOp] = nestedOps :+ op
  def initOpTypes: IndexedSeq[Type] = Extract.getAgg(this).initOpTypes.toFastSeq
  def seqOpTypes: IndexedSeq[Type] = Extract.getAgg(this).seqOpTypes.toFastSeq
  def emitResultType: EmitType = Extract.getAgg(this).resultEmitType
  def resultType: Type = emitResultType.virtualType
}

case class BasicPhysicalAggSig(override val op: AggOp, override val state: AggStateSig)
    extends PhysicalAggSig(op, state, Array())

case class GroupedAggSig(kt: VirtualTypeWithReq, nested: Seq[PhysicalAggSig])
    extends PhysicalAggSig(
      Group(),
      GroupedStateSig(kt, nested.map(_.state)),
      nested.flatMap(sig => sig.allOps).toArray,
    )

case class AggElementsAggSig(nested: Seq[PhysicalAggSig]) extends PhysicalAggSig(
      AggElements(),
      ArrayAggStateSig(nested.map(_.state)),
      nested.flatMap(sig => sig.allOps).toArray,
    )

case class ArrayLenAggSig(knownLength: Boolean, nested: Seq[PhysicalAggSig]) extends PhysicalAggSig(
      AggElementsLengthCheck(),
      ArrayAggStateSig(nested.map(_.state)),
      nested.flatMap(sig => sig.allOps).toArray,
    )

// The result of Extract
class Aggs(
  // The pre-extract IR
  original: IR,
  /* map each descendant of the original ir to its extract result (all contained aggs replaced with
   * their results) */
  rewriteMap: Memo[IR],
  // nodes which bind variables which are referenced in init op arguments
  bindingNodesReferenced: Memo[Unit],
  // The extracted void-typed initialization ir
  val init: IR,
  // The extracted void-typed update ir
  val seqPerElt: IR,
  // Must be bound to raw aggregators results in postAggIR
  val resultRef: Ref,
  // All (top-level) aggregators used
  val aggs: Array[PhysicalAggSig],
) {
  val states: Array[AggStateSig] = aggs.map(_.state)
  val nAggs: Int = aggs.length

  lazy val postAggIR: IR =
    rewriteMap.lookup(original)

  def rewriteFromInitBindingRoot(f: IR => IR): IR = {
    if (bindingNodesReferenced.m.isEmpty) {
      f(rewriteMap.lookup(original))
    } else {
      val irNumberMemo = Memo.empty[Int]
      var i = 0
      // find deepest binding node referenced
      // depth first search -- either DFS or BFS should work here given IR assumptions
      VisitIR(original) { x =>
        irNumberMemo.bind(x, i)
        i += 1
      }
      val rewriteRoot = bindingNodesReferenced.m.keys.maxBy(irNumberMemo.lookup)
      // only support let nodes here -- other binders like stream operators are undefined behavior
      RewriteTopDown.rewriteTopDown(
        original,
        {
          case ir if RefEquality(ir) == rewriteRoot =>
            val Block(bindings, body) = ir
            Block(bindings, f(rewriteMap.lookup(body)))
        },
      ).asInstanceOf[IR]
    }
  }

  def isCommutative: Boolean = {
    def aggCommutes(agg: PhysicalAggSig): Boolean = agg.allOps.forall(AggIsCommutative(_))
    aggs.forall(aggCommutes)
  }

  def shouldTreeAggregate: Boolean = {
    def containsBigAggregator(agg: PhysicalAggSig): Boolean = agg.allOps.exists {
      case AggElements() => true
      case AggElementsLengthCheck() => true
      case Downsample() => true
      case NDArraySum() => true
      case NDArrayMultiplyAdd() => true
      case Densify() => true
      case Group() => true
      case Take() => true
      case TakeBy(_) => true
      case Fold() => !agg.resultType.isPrimitive
      case CollectAsSet() => true
      case _ => false
    }
    aggs.exists(containsBigAggregator)
  }

  def deserialize(ctx: ExecuteContext, spec: BufferSpec)
    : ((HailClassLoader, HailTaskContext, Region, Array[Byte]) => Long) = {
    val (_, f) = ir.CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      states,
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      ir.DeserializeAggs(0, 0, spec, states),
    )

    val fsBc = ctx.fsBc;
    { (hcl: HailClassLoader, htc: HailTaskContext, aggRegion: Region, bytes: Array[Byte]) =>
      val f2 = f(hcl, fsBc.value, htc, aggRegion)
      f2.newAggState(aggRegion)
      f2.setSerializedAgg(0, bytes)
      f2(aggRegion)
      f2.getAggOffset()
    }
  }

  def serialize(ctx: ExecuteContext, spec: BufferSpec)
    : (HailClassLoader, HailTaskContext, Region, Long) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      states,
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      ir.SerializeAggs(0, 0, spec, states),
    )

    val fsBc = ctx.fsBc;
    { (hcl: HailClassLoader, htc: HailTaskContext, aggRegion: Region, off: Long) =>
      val f2 = f(hcl, fsBc.value, htc, aggRegion)
      f2.setAggState(aggRegion, off)
      f2(aggRegion)
      f2.storeAggsToRegion()
      f2.getSerializedAgg(0)
    }
  }

  def combOpFSerializedWorkersOnly(ctx: ExecuteContext, spec: BufferSpec)
    : (Array[Byte], Array[Byte]) => Array[Byte] = {
    combOpFSerializedFromRegionPool(ctx, spec) { () =>
      val htc = SparkTaskContext.get()
      val hcl = theHailClassLoaderForSparkWorkers
      if (htc == null) {
        throw new UnsupportedOperationException(
          s"Can't get htc. On worker = ${TaskContext.get != null}"
        )
      }
      (htc.getRegionPool(), hcl, htc)
    }
  }

  def combOpFSerializedFromRegionPool(ctx: ExecuteContext, spec: BufferSpec)
    : (() => (RegionPool, HailClassLoader, HailTaskContext)) => (
      (Array[Byte], Array[Byte]) => Array[Byte],
    ) = {
    val (_, f) = ir.CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      states ++ states,
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      Begin(FastSeq(
        ir.DeserializeAggs(0, 0, spec, states),
        ir.DeserializeAggs(nAggs, 1, spec, states),
        Begin(aggs.zipWithIndex.map { case (sig, i) => CombOp(i, i + nAggs, sig) }),
        SerializeAggs(0, 0, spec, states),
      )),
    )

    val fsBc = ctx.fsBc
    poolGetter: (() => (RegionPool, HailClassLoader, HailTaskContext)) => {
      (bytes1: Array[Byte], bytes2: Array[Byte]) =>
        val (pool, hcl, htc) = poolGetter()
        pool.scopedSmallRegion { r =>
          val f2 = f(hcl, fsBc.value, htc, r)
          f2.newAggState(r)
          f2.setSerializedAgg(0, bytes1)
          f2.setSerializedAgg(1, bytes2)
          f2(r)
          f2.storeAggsToRegion()
          f2.getSerializedAgg(0)
        }
    }
  }

  // Takes ownership of both input regions, and returns ownership of region in
  // resulting RegionValue.
  def combOpF(ctx: ExecuteContext, spec: BufferSpec)
    : (HailClassLoader, HailTaskContext, RegionValue, RegionValue) => RegionValue = {
    val fb = ir.EmitFunctionBuilder[AsmFunction4RegionLongRegionLongLong](
      ctx,
      "combOpF3",
      FastSeq[ParamType](classInfo[Region], LongInfo, classInfo[Region], LongInfo),
      LongInfo,
    )

    val leftAggRegion = fb.genFieldThisRef[Region]("agg_combine_left_top_region")
    val leftAggOff = fb.genFieldThisRef[Long]("agg_combine_left_off")
    val rightAggRegion = fb.genFieldThisRef[Region]("agg_combine_right_top_region")
    val rightAggOff = fb.genFieldThisRef[Long]("agg_combine_right_off")

    fb.emit(EmitCodeBuilder.scopedCode(fb.emb) { cb =>
      cb.assign(leftAggRegion, fb.getCodeParam[Region](1))
      cb.assign(leftAggOff, fb.getCodeParam[Long](2))
      cb.assign(rightAggRegion, fb.getCodeParam[Region](3))
      cb.assign(rightAggOff, fb.getCodeParam[Long](4))

      val leftStates = agg.StateTuple(states.map(s => AggStateSig.getState(s, fb.ecb)))
      val leftAggState = new agg.TupleAggregatorState(fb.ecb, leftStates, leftAggRegion, leftAggOff)
      val rightStates = agg.StateTuple(states.map(s => AggStateSig.getState(s, fb.ecb)))
      val rightAggState =
        new agg.TupleAggregatorState(fb.ecb, rightStates, rightAggRegion, rightAggOff)

      leftStates.createStates(cb)
      leftAggState.load(cb)

      rightStates.createStates(cb)
      rightAggState.load(cb)

      for (i <- 0 until nAggs) {
        val rvAgg = agg.Extract.getAgg(aggs(i))
        rvAgg.combOp(ctx, cb, leftAggRegion, leftAggState.states(i), rightAggState.states(i))
      }

      leftAggState.store(cb)
      rightAggState.store(cb)
      leftAggOff
    })

    val f = fb.resultWithIndex()
    val fsBc = ctx.fsBc

    { (hcl: HailClassLoader, htc: HailTaskContext, l: RegionValue, r: RegionValue) =>
      val comb = f(hcl, fsBc.value, htc, l.region)
      l.setOffset(comb(l.region, l.offset, r.region, r.offset))
      r.region.invalidate()
      l
    }
  }

  def results: IR =
    ResultOp.makeTuple(aggs)
}

object Extract {

  // All lets whose value depends on `name` (either directly or transitively through previous lets)
  // are returned in the first array, the rest are in the second.
  /* TODO: this is only being used to do ad hoc code motion. Remove when we have a real code motion
   * pass. */
  private def partitionDependentLets(lets: Array[(String, IR)], name: String)
    : (Array[(String, IR)], Array[(String, IR)]) = {
    val depBindings = mutable.HashSet.empty[String]
    depBindings += name

    val dep = new BoxedArrayBuilder[(String, IR)]
    val indep = new BoxedArrayBuilder[(String, IR)]

    lets.foreach { case x @ (name, value) =>
      if (value.typ == TVoid || value.isInstanceOf[ResultOp] || value.isInstanceOf[AggStateValue]) {
        /* if the value is side effecting, or implicitly reads the aggregator state, we can't lift
         * it */
        dep += x
      } else {
        val fv = FreeVariables(value, supportsAgg = false, supportsScan = false)
        if (fv.eval.m.keysIterator.exists(k => depBindings.contains(k))) {
          dep += x
          depBindings += name
        } else {
          indep += x
        }
      }
    }
    (dep.result(), indep.result())
  }

  // FIXME: move this to StagedAggregator?
  def getAgg(sig: PhysicalAggSig): StagedAggregator = sig match {
    case PhysicalAggSig(Sum(), TypedStateSig(t)) => new SumAggregator(t.t)
    case PhysicalAggSig(Product(), TypedStateSig(t)) => new ProductAggregator(t.t)
    case PhysicalAggSig(Min(), TypedStateSig(t)) => new MinAggregator(t.t)
    case PhysicalAggSig(Max(), TypedStateSig(t)) => new MaxAggregator(t.t)
    case PhysicalAggSig(PrevNonnull(), TypedStateSig(t)) => new PrevNonNullAggregator(t)
    case PhysicalAggSig(Count(), TypedStateSig(_)) => CountAggregator
    case PhysicalAggSig(Take(), TakeStateSig(t)) => new TakeAggregator(t)
    case PhysicalAggSig(TakeBy(_), TakeByStateSig(v, k, _)) => new TakeByAggregator(v, k)
    case PhysicalAggSig(ReservoirSample(), ReservoirSampleStateSig(t)) =>
      new ReservoirSampleAggregator(t)
    case PhysicalAggSig(Densify(), DensifyStateSig(v)) => new DensifyAggregator(v)
    case PhysicalAggSig(CallStats(), CallStatsStateSig()) => new CallStatsAggregator()
    case PhysicalAggSig(Collect(), CollectStateSig(t)) => new CollectAggregator(t)
    case PhysicalAggSig(CollectAsSet(), CollectAsSetStateSig(t)) => new CollectAsSetAggregator(t)
    case PhysicalAggSig(LinearRegression(), LinearRegressionStateSig()) =>
      new LinearRegressionAggregator()
    case PhysicalAggSig(ApproxCDF(), ApproxCDFStateSig()) => new ApproxCDFAggregator
    case PhysicalAggSig(Downsample(), DownsampleStateSig(labelType)) =>
      new DownsampleAggregator(labelType)
    case PhysicalAggSig(ImputeType(), ImputeTypeStateSig()) => new ImputeTypeAggregator()
    case ArrayLenAggSig(knownLength, nested) => // FIXME nested things shouldn't need to know state
      new ArrayElementLengthCheckAggregator(nested.map(getAgg).toArray, knownLength)
    case AggElementsAggSig(nested) =>
      new ArrayElementwiseOpAggregator(nested.map(getAgg).toArray)
    case GroupedAggSig(k, nested) =>
      new GroupedAggregator(k, nested.map(getAgg).toArray)
    case PhysicalAggSig(NDArraySum(), NDArraySumStateSig(nda)) =>
      new NDArraySumAggregator(nda)
    case PhysicalAggSig(NDArrayMultiplyAdd(), NDArrayMultiplyAddStateSig(nda)) =>
      new NDArrayMultiplyAddAggregator(nda)
    case PhysicalAggSig(Fold(), FoldStateSig(res, accumName, otherAccumName, combOpIR)) =>
      new FoldAggregator(res, accumName, otherAccumName, combOpIR)
  }

  def apply(ir: IR, r: RequirednessAnalysis, isScan: Boolean = false): Aggs = {
    val ab = new BoxedArrayBuilder[(InitOp, PhysicalAggSig)]()
    val seq = new BoxedArrayBuilder[(String, IR)]()
    val ref = Ref(genUID(), null)
    val memo = mutable.Map.empty[IR, Int]

    val bindingNodesReferenced = Memo.empty[Unit]
    val rewriteMap = Memo.empty[IR]
    extract(
      ir,
      BindingEnv.empty,
      bindingNodesReferenced,
      rewriteMap,
      ab,
      seq,
      memo,
      ref,
      r,
      isScan,
    )
    val (initOps, pAggSigs) = ab.result().unzip
    val rt = TTuple(initOps.map(_.aggSig.resultType): _*)
    ref._typ = rt

    new Aggs(
      ir,
      rewriteMap,
      bindingNodesReferenced,
      Begin(initOps),
      Let.void(seq.result()),
      ref,
      pAggSigs,
    )
  }

  private def extract(
    ir: IR,
    // bindings in scope for init op arguments
    env: BindingEnv[RefEquality[IR]],
    // nodes which bind variables which are referenced in init op arguments
    bindingNodesReferenced: Memo[Unit],
    /* map each desendant of the original ir to its extract result (all contained aggs replaced with
     * their results) */
    rewriteMap: Memo[IR],
    // set of contained aggs, and the init op for each
    ab: BoxedArrayBuilder[(InitOp, PhysicalAggSig)],
    // set of updates for contained aggs
    seqBuilder: BoxedArrayBuilder[(String, IR)],
    /* Map each contained ApplyAggOp, ApplyScanOp, or AggFold, to the index of the corresponding agg
     * state, used to perform CSE on agg ops */
    memo: mutable.Map[IR, Int],
    // a reference to the tuple of results of contained aggs
    result: IR,
    r: RequirednessAnalysis,
    isScan: Boolean,
  ): IR = {
    def newMemo: mutable.Map[IR, Int] = mutable.Map.empty[IR, Int]

    // For each free variable in each init op arg, add the binding site to bindingNodesReferenced
    def bindInitArgRefs(initArgs: IndexedSeq[IR]): Unit = {
      initArgs.foreach { arg =>
        val fv = FreeVariables(arg, false, false).eval
        fv.m.keys
          .flatMap(k => env.eval.lookupOption(k))
          .foreach(bindingNodesReferenced.bind(_, ()))
      }
    }

    val newNode = ir match {
      case x @ Block(bindings, body) =>
        var newEnv = env
        val newBindings = Array.newBuilder[Binding]
        newBindings.sizeHint(bindings)
        for (binding <- bindings) binding match {
          case Binding(name, value, Scope.EVAL) =>
            val newValue = this.extract(value, newEnv, bindingNodesReferenced, rewriteMap, ab,
              seqBuilder, memo, result, r, isScan)
            newBindings += Binding(name, newValue)
            newEnv = env.bindEval(name, RefEquality(x))
          case Binding(name, value, _) =>
            seqBuilder += name -> value
        }
        val newBody = this.extract(body, newEnv, bindingNodesReferenced, rewriteMap, ab, seqBuilder,
          memo, result, r, isScan)
        Block(newBindings.result(), newBody)
      case x: ApplyAggOp if !isScan =>
        val idx = memo.getOrElseUpdate(
          x, {
            val i = ab.length
            val op = x.aggSig.op
            bindInitArgRefs(x.initOpArgs)
            val state = PhysicalAggSig(op, AggStateSig(op, x.seqOpArgs, r))
            ab += InitOp(i, x.initOpArgs, state) -> state
            seqBuilder += "__void" -> SeqOp(i, x.seqOpArgs, state)
            i
          },
        )
        GetTupleElement(result, idx)
      case x: ApplyScanOp if isScan =>
        val idx = memo.getOrElseUpdate(
          x, {
            val i = ab.length
            val op = x.aggSig.op
            bindInitArgRefs(x.initOpArgs)
            val state = PhysicalAggSig(op, AggStateSig(op, x.seqOpArgs, r))
            ab += InitOp(i, x.initOpArgs, state) -> state
            seqBuilder += "__void" -> SeqOp(i, x.seqOpArgs, state)
            i
          },
        )
        GetTupleElement(result, idx)
      case x @ AggFold(zero, seqOp, combOp, accumName, otherAccumName, _) =>
        val idx = memo.getOrElseUpdate(
          x, {
            val i = ab.length
            val initOpArgs = IndexedSeq(zero)
            bindInitArgRefs(initOpArgs)
            val seqOpArgs = IndexedSeq(seqOp)
            val op = Fold()
            val resultEmitType = r(x).canonicalEmitType(x.typ)
            val foldStateSig = FoldStateSig(resultEmitType, accumName, otherAccumName, combOp)
            val signature = PhysicalAggSig(op, foldStateSig)
            ab += InitOp(i, initOpArgs, signature) -> signature
            // So seqOp has to be able to reference accumName.
            seqBuilder += accumName -> ResultOp(i, signature)
            seqBuilder += "__void" -> SeqOp(i, seqOpArgs, signature)
            i
          },
        )
        GetTupleElement(result, idx)
      case AggFilter(cond, aggIR, _) =>
        val newSeq = new BoxedArrayBuilder[(String, IR)]()
        val transformed = this.extract(aggIR, env, bindingNodesReferenced, rewriteMap, ab, newSeq,
          newMemo, result, r, isScan)

        seqBuilder += "__void" -> If(cond, Let.void(newSeq.result()), Void())
        transformed

      case AggExplode(array, name, aggBody, _) =>
        val newSeq = new BoxedArrayBuilder[(String, IR)]()
        val transformed = this.extract(aggBody, env, bindingNodesReferenced, rewriteMap, ab, newSeq,
          newMemo, result, r, isScan)

        val (dependent, independent) = partitionDependentLets(newSeq.result(), name)
        seqBuilder ++= independent
        seqBuilder += "__void" -> StreamFor(array, name, Let.void(dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        val newAggs = new BoxedArrayBuilder[(InitOp, PhysicalAggSig)]()
        val newSeq = new BoxedArrayBuilder[(String, IR)]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(
          aggIR,
          env,
          bindingNodesReferenced,
          rewriteMap,
          newAggs,
          newSeq,
          newMemo,
          GetField(newRef, "value"),
          r,
          isScan,
        )

        val i = ab.length
        val (initOps, pAggSigs) = newAggs.result().unzip

        val rt = TDict(key.typ, TTuple(initOps.map(_.aggSig.resultType): _*))
        newRef._typ = rt.elementType

        val groupState = AggStateSig.grouped(key, pAggSigs.map(_.state), r)
        val groupSig = GroupedAggSig(groupState.kt, pAggSigs.toFastSeq)
        ab += InitOp(i, FastSeq(Begin(initOps)), groupSig) -> groupSig
        seqBuilder += "__void" -> SeqOp(
          i,
          FastSeq(key, Let.void(newSeq.result())),
          groupSig,
        )

        ToDict(StreamMap(
          ToStream(GetTupleElement(result, i)),
          newRef.name,
          MakeTuple.ordered(FastSeq(GetField(newRef, "key"), transformed)),
        ))

      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, _) =>
        val newAggs = new BoxedArrayBuilder[(InitOp, PhysicalAggSig)]()
        val newSeq = new BoxedArrayBuilder[(String, IR)]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggBody, env, bindingNodesReferenced, rewriteMap, newAggs,
          newSeq, newMemo, newRef, r, isScan)

        val (dependent, independent) = partitionDependentLets(newSeq.result(), elementName)

        val i = ab.length
        val (initOps, pAggSigs) = newAggs.result().unzip

        val rt = TArray(TTuple(initOps.map(_.aggSig.resultType): _*))
        newRef._typ = rt.elementType

        val checkSig = ArrayLenAggSig(knownLength.isDefined, pAggSigs)
        val eltSig = AggElementsAggSig(pAggSigs)

        val aRef = Ref(genUID(), a.typ)

        ab += InitOp(
          i,
          knownLength.map(FastSeq(_)).getOrElse(FastSeq[IR]()) :+ Begin(initOps),
          checkSig,
        ) -> checkSig

        seqBuilder ++= independent
        seqBuilder += aRef.name -> a
        seqBuilder += "__void" -> SeqOp(i, FastSeq(ArrayLen(aRef)), checkSig)
        seqBuilder +=
          "__void" -> StreamFor(
            StreamRange(I32(0), ArrayLen(aRef), I32(1)),
            indexName,
            SeqOp(
              i,
              FastSeq(
                Ref(indexName, TInt32),
                Let.void((elementName, ArrayRef(aRef, Ref(indexName, TInt32))) +: dependent),
              ),
              eltSig,
            ),
          )

        val rUID = Ref(genUID(), rt)
        Let(
          FastSeq(rUID.name -> GetTupleElement(result, i)),
          ToArray(StreamMap(
            StreamRange(0, ArrayLen(rUID), 1),
            indexName,
            Let(
              FastSeq(newRef.name -> ArrayRef(rUID, Ref(indexName, TInt32))),
              transformed,
            ),
          )),
        )

      case x: StreamAgg =>
        assert(!ContainsScan(x))
        x
      case x: StreamAggScan =>
        assert(!ContainsAgg(x))
        x
      case x =>
        x.mapChildrenWithIndex { case (child: IR, i) =>
          val newEnv = env.extend(Bindings.get(x, i).map((_, _) => RefEquality(x)))

          this.extract(child, newEnv, bindingNodesReferenced, rewriteMap, ab, seqBuilder, memo,
            result, r, isScan)
        }
    }

    rewriteMap.bind(ir, newNode)
    newNode
  }
}
