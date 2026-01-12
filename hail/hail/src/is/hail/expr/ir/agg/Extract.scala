package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionPool, RegionValue}
import is.hail.asm4s._
import is.hail.backend.{ExecuteContext, HailTaskContext}
import is.hail.backend.spark.SparkTaskContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.compat.mutable.Growable
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.ir.compile.CompileWithAggregators
import is.hail.expr.ir.defs._
import is.hail.io.BufferSpec
import is.hail.types.{tcoerce, TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.types.physical.stypes.EmitType
import is.hail.types.virtual._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

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

  def grouped(k: IR, aggs: IndexedSeq[AggStateSig], r: RequirednessAnalysis): GroupedStateSig =
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
        StateTuple(nested.map(sig => AggStateSig.getState(sig, cb))),
      )
    case GroupedStateSig(kt, nested) =>
      new DictState(cb, kt, StateTuple(nested.map(sig => AggStateSig.getState(sig, cb))))
    case NDArraySumStateSig(nda) => new TypedRegionBackedAggState(nda, cb)
    case NDArrayMultiplyAddStateSig(nda) =>
      new TypedRegionBackedAggState(nda, cb)
    case FoldStateSig(resultEmitType, _, _, _) =>
      val vWithReq = resultEmitType.typeWithRequiredness
      new TypedRegionBackedAggState(vWithReq, cb)
    case LinearRegressionStateSig() => new LinearRegressionAggregatorState(cb)
  }
}

sealed abstract class AggStateSig(
  val t: IndexedSeq[VirtualTypeWithReq],
  val n: Option[IndexedSeq[AggStateSig]],
)

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

case class ArrayAggStateSig(nested: IndexedSeq[AggStateSig])
    extends AggStateSig(Array[VirtualTypeWithReq](), Some(nested))

case class GroupedStateSig(kt: VirtualTypeWithReq, nested: IndexedSeq[AggStateSig])
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
  accumName: Name,
  otherAccumName: Name,
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

case class GroupedAggSig(kt: VirtualTypeWithReq, nested: IndexedSeq[PhysicalAggSig])
    extends PhysicalAggSig(
      Group(),
      GroupedStateSig(kt, nested.map(_.state)),
      nested.flatMap(sig => sig.allOps).toArray,
    )

case class AggElementsAggSig(nested: IndexedSeq[PhysicalAggSig]) extends PhysicalAggSig(
      AggElements(),
      ArrayAggStateSig(nested.map(_.state)),
      nested.flatMap(sig => sig.allOps).toArray,
    )

case class ArrayLenAggSig(knownLength: Boolean, nested: IndexedSeq[PhysicalAggSig])
    extends PhysicalAggSig(
      AggElementsLengthCheck(),
      ArrayAggStateSig(nested.map(_.state)),
      nested.flatMap(sig => sig.allOps).toArray,
    )

class AggSignatures(val sigs: IndexedSeq[PhysicalAggSig]) {
  val nAggs: Int = sigs.length
  def isEmpty: Boolean = sigs.isEmpty
  val states: IndexedSeq[AggStateSig] = sigs.map(_.state)

  def resultsOp: IR = ResultOp.makeTuple(sigs)

  def valuesOp: IR =
    MakeTuple.ordered(states.zipWithIndex.map { case (state, i) =>
      AggStateValue(i, state)
    })

  def initFromSerializedValueOp(statesValue: TrivialIR): IR =
    Begin(states.zipWithIndex.map { case (state, i) =>
      InitFromSerializedValue(i, GetTupleElement(statesValue, i), state)
    })

  def combOpValues(values: TrivialIR): IR =
    Begin(sigs.zipWithIndex.map { case (sig, i) =>
      CombOpValue(i, GetTupleElement(values, i), sig)
    })

  def isCommutative: Boolean = {
    def aggCommutes(agg: PhysicalAggSig): Boolean = agg.allOps.forall(AggIsCommutative(_))
    sigs.forall(aggCommutes)
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
    sigs.exists(containsBigAggregator)
  }

  def deserialize(ctx: ExecuteContext, spec: BufferSpec)
    : ((HailClassLoader, HailTaskContext, Region, Array[Byte]) => Long) = {
    val (_, f) = CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      states,
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      DeserializeAggs(0, 0, spec, states),
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
    val (_, f) = CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      states,
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      SerializeAggs(0, 0, spec, states),
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
          s"Can't get htc. On worker = ${TaskContext.get() != null}"
        )
      }
      (htc.getRegionPool(), hcl, htc)
    }
  }

  def combOpFSerializedFromRegionPool(ctx: ExecuteContext, spec: BufferSpec)
    : (() => (RegionPool, HailClassLoader, HailTaskContext)) => (
      (Array[Byte], Array[Byte]) => Array[Byte],
    ) = {
    val (_, f) = CompileWithAggregators[AsmFunction1RegionUnit](
      ctx,
      states ++ states,
      FastSeq(),
      FastSeq(classInfo[Region]),
      UnitInfo,
      Begin(FastSeq(
        DeserializeAggs(0, 0, spec, states),
        DeserializeAggs(nAggs, 1, spec, states),
        Begin(sigs.zipWithIndex.map { case (sig, i) => CombOp(i, i + nAggs, sig) }),
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

      val leftStates = StateTuple(states.map(s => AggStateSig.getState(s, fb.ecb)))
      val leftAggState = new TupleAggregatorState(fb.ecb, leftStates, leftAggRegion, leftAggOff)
      val rightStates = StateTuple(states.map(s => AggStateSig.getState(s, fb.ecb)))
      val rightAggState =
        new TupleAggregatorState(fb.ecb, rightStates, rightAggRegion, rightAggOff)

      leftStates.createStates(cb)
      leftAggState.load(cb)

      rightStates.createStates(cb)
      rightAggState.load(cb)

      for (i <- 0 until nAggs) {
        val rvAgg = Extract.getAgg(sigs(i))
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
}

// The result of Extract
class ExtractedAggs(
  ctx: ExecuteContext,
  // Must be bound in `init` and `postAggIR`
  val initBindings: IndexedSeq[(Name, IR)],
  // The extracted void-typed initialization ir
  val init: IR,
  // The extracted void-typed update ir
  val seqPerElt: IR,
  // The result IR, which references `resultRef`
  val result: IR,
  // All (top-level) aggregators used
  val sigs: AggSignatures,
) {
  def independent: IndependentExtractedAggs = new IndependentExtractedAggs(
    ForwardLets(ctx, Let(initBindings, init)),
    ForwardLets(ctx, seqPerElt),
    ForwardLets(ctx, Let(initBindings, result)),
    sigs,
  )
}

class IndependentExtractedAggs(
  val init: IR,
  val seqPerElt: IR,
  val result: IR,
  val sigs: AggSignatures,
)

object Extract {

  // All lets whose value depends on `name` (either directly or transitively through previous lets)
  // are returned in the first array, the rest are in the second.
  /* TODO: this is only being used to do ad hoc code motion. Remove when we have a real code motion
   * pass. */
  private def partitionDependentLets(lets: IndexedSeq[(Name, IR)], name: Name)
    : (IndexedSeq[(Name, IR)], IndexedSeq[(Name, IR)]) = {
    val depBindings = mutable.HashSet.empty[Name]
    depBindings += name

    val dep = ArraySeq.newBuilder[(Name, IR)]
    val indep = ArraySeq.newBuilder[(Name, IR)]

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

  def apply(ctx: ExecuteContext, ir: IR, r: RequirednessAnalysis, isScan: Boolean = false)
    : ExtractedAggs = {
    val initBindings = ArraySeq.newBuilder[(Name, IR)]
    val initBuilder = ArrayBuffer.empty[InitOp]
    val seqBuilder = ArraySeq.newBuilder[(Name, IR)]
    val memo = mutable.Map.empty[IR, Int]
    val result = Ref(freshName(), null)

    val postAggIR = extract(
      ir,
      BindingEnv.empty,
      initBindings,
      initBuilder,
      seqBuilder,
      memo,
      result,
      r,
      isScan,
    )

    val initOps = initBuilder.to(ArraySeq)
    val pAggSigs = initOps.map(_.aggSig)
    val sigs = new AggSignatures(pAggSigs)
    result._typ = sigs.resultsOp.typ

    new ExtractedAggs(
      ctx,
      initBindings.result(),
      Begin(initOps),
      Let.void(seqBuilder.result()),
      Let(FastSeq(result.name -> sigs.resultsOp), postAggIR),
      new AggSignatures(pAggSigs),
    )
  }

  private class BindingState {
    var usedInInitOp: Boolean = false
  }

  private def extract(
    ir: IR,
    env: BindingEnv[BindingState],
    // Bindings in scope for init op arguments. Will also be in scope in post-agg IR.
    initBindings: Growable[(Name, IR)],
    // set of contained aggs, and the init op for each
    initBuilder: mutable.Buffer[InitOp],
    /* Set of updates for contained aggs, with intermediate let-bound values. Will be wrapped in a
     * Block. */
    seqBuilder: Growable[(Name, IR)],
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
        val fv = FreeVariables(arg, supportsAgg = false, supportsScan = false).eval
        fv.m.keys
          .flatMap(k => env.eval.lookupOption(k))
          .foreach(_.usedInInitOp = true)
      }
    }

    ir match {
      case Block(bindings, body) =>
        var newEnv = env
        val bindingsTemp = Array.newBuilder[(Name, IR)]
        bindingsTemp.sizeHint(bindings)
        for (binding <- bindings) binding match {
          case Binding(name, value, Scope.EVAL) =>
            val newValue =
              this.extract(value, newEnv, initBindings, initBuilder, seqBuilder, memo, result, r,
                isScan)
            bindingsTemp += name -> newValue
            newEnv = newEnv.bindEval(name, new BindingState)
          case Binding(name, value, _) =>
            seqBuilder += name -> value
        }
        val newBody = this.extract(body, newEnv, initBindings, initBuilder, seqBuilder,
          memo, result, r, isScan)
        val newBindings = bindingsTemp.result()
        bindingsTemp.clear()
        newBindings.foreach { case b @ (name, f) =>
          if (newEnv.eval(name).usedInInitOp) initBindings += ((name, f))
          else bindingsTemp += b
        }

        Block(
          bindingsTemp.result().map { case (name, value) => Binding(name, value) },
          newBody,
        )

      case x: ApplyAggOp if !isScan =>
        val idx = memo.getOrElseUpdate(
          x, {
            val i = initBuilder.length
            bindInitArgRefs(x.initOpArgs)
            val op = x.op
            val state = PhysicalAggSig(op, AggStateSig(op, x.seqOpArgs, r))
            initBuilder += InitOp(i, x.initOpArgs, state)
            seqBuilder += freshName() -> SeqOp(i, x.seqOpArgs, state)
            i
          },
        )

        GetTupleElement(result, idx)

      case x: ApplyScanOp if isScan =>
        val idx = memo.getOrElseUpdate(
          x, {
            val i = initBuilder.length
            bindInitArgRefs(x.initOpArgs)
            val op = x.op
            val state = PhysicalAggSig(op, AggStateSig(op, x.seqOpArgs, r))
            initBuilder += InitOp(i, x.initOpArgs, state)
            seqBuilder += freshName() -> SeqOp(i, x.seqOpArgs, state)
            i
          },
        )

        GetTupleElement(result, idx)

      case x @ AggFold(zero, seqOp, combOp, accumName, otherAccumName, _) =>
        val idx = memo.getOrElseUpdate(
          x, {
            val i = initBuilder.length
            val initOpArgs = IndexedSeq(zero)
            bindInitArgRefs(initOpArgs)
            val seqOpArgs = IndexedSeq(seqOp)
            val op = Fold()
            val resultEmitType = r(x).canonicalEmitType(x.typ)
            val foldStateSig = FoldStateSig(resultEmitType, accumName, otherAccumName, combOp)
            val state = PhysicalAggSig(op, foldStateSig)
            initBuilder += InitOp(i, initOpArgs, state)
            // So seqOp has to be able to reference accumName.
            seqBuilder += accumName -> ResultOp(i, state)
            seqBuilder += freshName() -> SeqOp(i, seqOpArgs, state)
            i
          },
        )

        GetTupleElement(result, idx)

      case AggFilter(cond, aggIR, _) =>
        val newSeq = ArraySeq.newBuilder[(Name, IR)]
        val transformed = this.extract(aggIR, env, initBindings, initBuilder, newSeq,
          newMemo, result, r, isScan)

        seqBuilder += freshName() -> If(cond, Let.void(newSeq.result()), Void())
        transformed

      case AggExplode(array, name, aggBody, _) =>
        val newSeq = ArraySeq.newBuilder[(Name, IR)]
        val transformed = this.extract(aggBody, env, initBindings, initBuilder, newSeq,
          newMemo, result, r, isScan)

        val (dependent, independent) = partitionDependentLets(newSeq.result(), name)
        seqBuilder ++= independent
        seqBuilder += freshName() -> StreamFor(array, name, Let.void(dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        val i = initBuilder.length
        val newInit = ArrayBuffer.empty[InitOp]
        val newSeq = ArraySeq.newBuilder[(Name, IR)]
        val newRef = Ref(freshName(), null)
        val transformed = this.extract(
          aggIR,
          env,
          initBindings,
          newInit,
          newSeq,
          newMemo,
          GetField(newRef, "value"),
          r,
          isScan,
        )
        val initOps = newInit.to(ArraySeq)

        val pAggSigs = initOps.map(_.aggSig)
        val groupState = AggStateSig.grouped(key, pAggSigs.map(_.state), r)
        val groupSig = GroupedAggSig(groupState.kt, pAggSigs.toFastSeq)
        initBuilder += InitOp(i, FastSeq(Begin(initOps)), groupSig)
        seqBuilder += freshName() -> SeqOp(
          i,
          FastSeq(key, Let.void(newSeq.result())),
          groupSig,
        )

        val rt = tcoerce[TDict](groupSig.resultType)
        newRef._typ = rt.elementType

        ToDict(StreamMap(
          ToStream(GetTupleElement(result, i)),
          newRef.name,
          MakeTuple.ordered(FastSeq(GetField(newRef, "key"), transformed)),
        ))

      case x @ AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, _) =>
        val i = initBuilder.length
        val newAggs = ArrayBuffer.empty[InitOp]
        val newSeq = ArraySeq.newBuilder[(Name, IR)]
        val newRef = Ref(freshName(), null)

        val transformed = this.extract(aggBody, env, initBindings, newAggs, newSeq,
          newMemo, newRef, r, isScan)

        val initOps = newAggs.to(ArraySeq)
        val pAggSigs = initOps.map(_.aggSig)
        val checkSig = ArrayLenAggSig(x.knownLength.isDefined, pAggSigs)
        val nestedSigs = checkSig.nested
        val rt = TArray(TTuple(nestedSigs.map(_.resultType): _*))
        newRef._typ = rt.elementType

        val (dependent, independent) = partitionDependentLets(newSeq.result(), elementName)

        val eltSig = AggElementsAggSig(nestedSigs)

        val aRef = Ref(freshName(), a.typ)

        initBuilder += InitOp(
          i,
          knownLength.map(FastSeq(_)).getOrElse(FastSeq[IR]()) :+ Begin(initOps),
          checkSig,
        )

        seqBuilder ++= independent
        seqBuilder += aRef.name -> a
        seqBuilder += freshName() -> SeqOp(i, FastSeq(ArrayLen(aRef)), checkSig)
        seqBuilder +=
          freshName() -> StreamFor(
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

        val rUID = Ref(freshName(), rt)

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
          val newEnv = env.extend(Bindings.get(x, i).map((_, _) => new BindingState))

          this.extract(child, newEnv, initBindings, initBuilder, seqBuilder, memo,
            result, r, isScan)
        }
    }
  }
}
