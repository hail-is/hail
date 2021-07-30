package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionPool, RegionValue}
import is.hail.asm4s._
import is.hail.backend.spark.SparkTaskContext
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.io.BufferSpec
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.types.{TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.utils._
import org.apache.spark.TaskContext

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

class UnsupportedExtraction(msg: String) extends Exception(msg)

object AggStateSig {
  def apply(op: AggOp, initOpArgs: Seq[IR], seqOpArgs: Seq[IR], r: RequirednessAnalysis): AggStateSig = {
    val inits = initOpArgs.map { i => i -> (if (i.typ == TVoid) null else r(i)) }
    val seqs = seqOpArgs.map { s => s -> (if (s.typ == TVoid) null else r(s)) }
    apply(op, inits, seqs)
  }
  def apply(op: AggOp, initOpArgs: Seq[(IR, TypeWithRequiredness)], seqOpArgs: Seq[(IR, TypeWithRequiredness)]): AggStateSig = {
    val seqVTypes = seqOpArgs.map { case (a, r) => VirtualTypeWithReq(a.typ, r) }
    op match {
      case Sum() | Product() => TypedStateSig(seqVTypes.head.setRequired(true))
      case Min() | Max()  => TypedStateSig(seqVTypes.head.setRequired(false))
      case Count() => TypedStateSig(VirtualTypeWithReq.fullyOptional(TInt64).setRequired(true))
      case Take() => TakeStateSig(seqVTypes.head)
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
      case NDArraySum() => NDArraySumStateSig(seqVTypes.head.setRequired(false)) // set required to false to handle empty aggs
      case _ => throw new UnsupportedExtraction(op.toString)
    }
  }
  def grouped(k: IR, aggs: Seq[AggStateSig], r: RequirednessAnalysis): GroupedStateSig =
    GroupedStateSig(VirtualTypeWithReq(k.typ, r(k)), aggs)
  def arrayelements(aggs: Seq[AggStateSig]): ArrayAggStateSig =
    ArrayAggStateSig(aggs)

  def getState(sig: AggStateSig, cb: EmitClassBuilder[_]): AggregatorState = sig match {
    case TypedStateSig(vt) if vt.t.isPrimitive => new PrimitiveRVAState(Array(vt), cb)
    case TypedStateSig(vt) => new TypedRegionBackedAggState(vt, cb)
    case DownsampleStateSig(labelType) => new DownsampleState(cb, labelType)
    case TakeStateSig(vt) => new TakeRVAS(vt, cb)
    case DensifyStateSig(vt) => new DensifyState(vt, cb)
    case TakeByStateSig(vt, kt, so) => new TakeByRVAS(vt, kt, cb, so)
    case CollectStateSig(pt) => new CollectAggState(pt, cb)
    case CollectAsSetStateSig(pt) => new AppendOnlySetState(cb, pt)
    case CallStatsStateSig() => new CallStatsState(cb)
    case ApproxCDFStateSig() => new ApproxCDFState(cb)
    case ImputeTypeStateSig() => new ImputeTypeState(cb)
    case ArrayAggStateSig(nested) => new ArrayElementState(cb, StateTuple(nested.map(sig => AggStateSig.getState(sig, cb)).toArray))
    case GroupedStateSig(kt, nested) => new DictState(cb, kt, StateTuple(nested.map(sig => AggStateSig.getState(sig, cb)).toArray))
    case NDArraySumStateSig(nda) => new TypedRegionBackedAggState(nda, cb)
    case LinearRegressionStateSig() => new LinearRegressionAggregatorState(cb)
  }
}

sealed abstract class AggStateSig(val t: Seq[VirtualTypeWithReq], val n: Option[Seq[AggStateSig]])
case class TypedStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class DownsampleStateSig(labelType: VirtualTypeWithReq) extends AggStateSig(Array(labelType), None)
case class TakeStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class TakeByStateSig(vt: VirtualTypeWithReq, kt: VirtualTypeWithReq, so: SortOrder) extends AggStateSig(Array(vt, kt), None)
case class DensifyStateSig(vt: VirtualTypeWithReq) extends AggStateSig(Array(vt), None)
case class CollectStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class CollectAsSetStateSig(pt: VirtualTypeWithReq) extends AggStateSig(Array(pt), None)
case class CallStatsStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)
case class ImputeTypeStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)
case class ArrayAggStateSig(nested: Seq[AggStateSig]) extends AggStateSig(Array[VirtualTypeWithReq](), Some(nested))
case class GroupedStateSig(kt: VirtualTypeWithReq, nested: Seq[AggStateSig]) extends AggStateSig(Array(kt), Some(nested))
case class ApproxCDFStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)
case class LinearRegressionStateSig() extends AggStateSig(Array[VirtualTypeWithReq](), None)
case class NDArraySumStateSig(nda: VirtualTypeWithReq) extends AggStateSig(Array[VirtualTypeWithReq](nda), None) {
  require(!nda.r.required)
}

object PhysicalAggSig {
  def apply(op: AggOp, state: AggStateSig): PhysicalAggSig = BasicPhysicalAggSig(op, state)
  def unapply(v: PhysicalAggSig): Option[(AggOp, AggStateSig)] = if (v.nestedOps.isEmpty) Some(v.op -> v.state) else None
}

class PhysicalAggSig(val op: AggOp, val state: AggStateSig, val nestedOps: Array[AggOp]) {
  val allOps: Array[AggOp] = nestedOps :+ op
  def initOpTypes: IndexedSeq[Type] = Extract.getAgg(this).initOpTypes.toFastIndexedSeq
  def seqOpTypes: IndexedSeq[Type] = Extract.getAgg(this).seqOpTypes.toFastIndexedSeq
  def pResultType: PType = Extract.getAgg(this).resultType
  def resultType: Type = pResultType.virtualType
}

case class BasicPhysicalAggSig(override val op: AggOp, override val state: AggStateSig) extends PhysicalAggSig(op, state, Array())

case class GroupedAggSig(kt: VirtualTypeWithReq, nested: Seq[PhysicalAggSig]) extends
  PhysicalAggSig(Group(), GroupedStateSig(kt, nested.map(_.state)), nested.flatMap(sig => sig.allOps).toArray)
case class AggElementsAggSig(nested: Seq[PhysicalAggSig]) extends
  PhysicalAggSig(AggElements(), ArrayAggStateSig(nested.map(_.state)), nested.flatMap(sig => sig.allOps).toArray)

case class ArrayLenAggSig(knownLength: Boolean, nested: Seq[PhysicalAggSig]) extends
  PhysicalAggSig(AggElementsLengthCheck(), ArrayAggStateSig(nested.map(_.state)), nested.flatMap(sig => sig.allOps).toArray)

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[PhysicalAggSig]) {
  val states: Array[AggStateSig] = aggs.map(_.state)
  val nAggs: Int = aggs.length

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
      case Densify() => true
      case _ => false
    }
    aggs.exists(containsBigAggregator)
  }

  def eltOp(ctx: ExecuteContext): IR = seqPerElt

  def deserialize(ctx: ExecuteContext, spec: BufferSpec): ((Region, Array[Byte]) => Long) = {
    val (_, f) = ir.CompileWithAggregators[AsmFunction1RegionUnit](ctx,
      states,
      FastIndexedSeq(),
      FastIndexedSeq(classInfo[Region]), UnitInfo,
      ir.DeserializeAggs(0, 0, spec, states))

    val fsBc = ctx.fsBc;
    { (aggRegion: Region, bytes: Array[Byte]) =>
      val f2 = f(fsBc.value, 0, aggRegion)
      f2.newAggState(aggRegion)
      f2.setSerializedAgg(0, bytes)
      f2(aggRegion)
      f2.getAggOffset()
    }
  }

  def serialize(ctx: ExecuteContext, spec: BufferSpec): (Region, Long) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators[AsmFunction1RegionUnit](ctx,
      states,
      FastIndexedSeq(),
      FastIndexedSeq(classInfo[Region]), UnitInfo,
      ir.SerializeAggs(0, 0, spec, states))

    val fsBc = ctx.fsBc;
    { (aggRegion: Region, off: Long) =>
      val f2 = f(fsBc.value, 0, aggRegion)
      f2.setAggState(aggRegion, off)
      f2(aggRegion)
      f2.storeAggsToRegion()
      f2.getSerializedAgg(0)
    }
  }

  def combOpFSerializedWorkersOnly(ctx: ExecuteContext, spec: BufferSpec): (Array[Byte], Array[Byte]) => Array[Byte] = {
    combOpFSerializedFromRegionPool(ctx, spec)(() => {
      val htc = SparkTaskContext.get()
      if (htc == null) {
        throw new UnsupportedOperationException(s"Can't get htc. On worker = ${TaskContext.get != null}")
      }
      htc.getRegionPool()
    })
  }

  def combOpFSerializedFromRegionPool(ctx: ExecuteContext, spec: BufferSpec): (() => RegionPool) => ((Array[Byte], Array[Byte]) => Array[Byte]) = {
    val (_, f) = ir.CompileWithAggregators[AsmFunction1RegionUnit](ctx,
      states ++ states,
      FastIndexedSeq(),
      FastIndexedSeq(classInfo[Region]), UnitInfo,
      Begin(FastSeq(
        ir.DeserializeAggs(0, 0, spec, states),
        ir.DeserializeAggs(nAggs, 1, spec, states),
        Begin(aggs.zipWithIndex.map { case (sig, i) => CombOp(i, i + nAggs, sig) }),
        SerializeAggs(0, 0, spec, states)
      )))

    val fsBc = ctx.fsBc
    poolGetter: (() => RegionPool) => { (bytes1: Array[Byte], bytes2: Array[Byte]) =>
      poolGetter().scopedSmallRegion { r =>
        val f2 = f(fsBc.value, 0, r)
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
  def combOpF(ctx: ExecuteContext, spec: BufferSpec): (RegionValue, RegionValue) => RegionValue = {
    val fb = ir.EmitFunctionBuilder[AsmFunction4RegionLongRegionLongLong](
      ctx,
      "combOpF3",
      FastIndexedSeq[ParamType](classInfo[Region], LongInfo, classInfo[Region], LongInfo),
      LongInfo)

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
      val rightAggState = new agg.TupleAggregatorState(fb.ecb, rightStates, rightAggRegion, rightAggOff)

      leftStates.createStates(cb)
      leftAggState.load(cb)

      rightStates.createStates(cb)
      rightAggState.load(cb)

      for (i <- 0 until nAggs) {
        val rvAgg = agg.Extract.getAgg(aggs(i))
        rvAgg.combOp(cb, leftAggState.states(i), rightAggState.states(i))
      }

      leftAggState.store(cb)
      rightAggState.store(cb)
      leftAggOff
    })

    val f = fb.resultWithIndex()
    val fsBc = ctx.fsBc

    { (l: RegionValue, r: RegionValue) =>
      val comb = f(fsBc.value, 0, l.region)
      l.setOffset(comb(l.region, l.offset, r.region, r.offset))
      r.region.invalidate()
      l
    }
  }

  def results: IR = ResultOp(0, aggs)
}

object Extract {

  def partitionDependentLets(lets: Array[AggLet], name: String): (Array[AggLet], Array[AggLet]) = {
    val depBindings = mutable.HashSet.empty[String]
    depBindings += name

    val dep = new BoxedArrayBuilder[AggLet]
    val indep = new BoxedArrayBuilder[AggLet]

    lets.foreach { l =>
      val fv = FreeVariables(l.value, supportsAgg = false, supportsScan = false)
      if (fv.eval.m.keysIterator.exists(k => depBindings.contains(k))) {
        dep += l
        depBindings += l.name
      } else
        indep += l
    }
    (dep.result(), indep.result())
  }

  def addLets(ir: IR, lets: Array[AggLet]): IR = {
    assert(lets.areDistinct())
    lets.foldRight[IR](ir) { case (al, comb) => Let(al.name, al.value, comb) }
  }

  def getResultType(aggSig: AggSignature): Type = aggSig match {
    case AggSignature(Sum(), _, Seq(t)) => t
    case AggSignature(Product(), _, Seq(t)) => t
    case AggSignature(Min(), _, Seq(t)) => t
    case AggSignature(Max(), _, Seq(t)) => t
    case AggSignature(Count(), _, _) => TInt64
    case AggSignature(Take(), _, Seq(t)) => TArray(t)
    case AggSignature(CallStats(), _, _) => CallStatsState.resultType.virtualType
    case AggSignature(TakeBy(_), _, Seq(value, key)) => TArray(value)
    case AggSignature(PrevNonnull(), _, Seq(t)) => t
    case AggSignature(CollectAsSet(), _, Seq(t)) => TSet(t)
    case AggSignature(Collect(), _, Seq(t)) => TArray(t)
    case AggSignature(Densify(), _, Seq(t)) => t
    case AggSignature(ImputeType(), _, _) => ImputeTypeState.resultType.virtualType
    case AggSignature(LinearRegression(), _, _) =>
      LinearRegressionAggregator.resultType.virtualType
    case AggSignature(ApproxCDF(), _, _) => QuantilesAggregator.resultType.virtualType
    case AggSignature(Downsample(), _, Seq(_, _, label)) => DownsampleAggregator.resultType
    case AggSignature(NDArraySum(), _, Seq(t)) => t
    case _ => throw new UnsupportedExtraction(aggSig.toString)  }

  def getAgg(sig: PhysicalAggSig): StagedAggregator = sig match {
    case PhysicalAggSig(Sum(), TypedStateSig(t)) => new SumAggregator(t.t)
    case PhysicalAggSig(Product(), TypedStateSig(t)) => new ProductAggregator(t.t)
    case PhysicalAggSig(Min(), TypedStateSig(t)) => new MinAggregator(t.t)
    case PhysicalAggSig(Max(), TypedStateSig(t)) => new MaxAggregator(t.t)
    case PhysicalAggSig(PrevNonnull(), TypedStateSig(t)) => new PrevNonNullAggregator(t)
    case PhysicalAggSig(Count(), TypedStateSig(_)) => CountAggregator
    case PhysicalAggSig(Take(), TakeStateSig(t)) => new TakeAggregator(t)
    case PhysicalAggSig(TakeBy(_), TakeByStateSig(v, k, _)) => new TakeByAggregator(v, k)
    case PhysicalAggSig(Densify(), DensifyStateSig(v)) => new DensifyAggregator(v)
    case PhysicalAggSig(CallStats(), CallStatsStateSig()) => new CallStatsAggregator()
    case PhysicalAggSig(Collect(), CollectStateSig(t)) => new CollectAggregator(t)
    case PhysicalAggSig(CollectAsSet(), CollectAsSetStateSig(t)) => new CollectAsSetAggregator(t)
    case PhysicalAggSig(LinearRegression(), LinearRegressionStateSig()) => new LinearRegressionAggregator()
    case PhysicalAggSig(ApproxCDF(), ApproxCDFStateSig()) => new ApproxCDFAggregator
    case PhysicalAggSig(Downsample(), DownsampleStateSig(labelType)) => new DownsampleAggregator(labelType)
    case PhysicalAggSig(ImputeType(), ImputeTypeStateSig()) => new ImputeTypeAggregator()
    case ArrayLenAggSig(knownLength, nested) => //FIXME nested things shouldn't need to know state
      new ArrayElementLengthCheckAggregator(nested.map(getAgg).toArray, knownLength)
    case AggElementsAggSig(nested) =>
      new ArrayElementwiseOpAggregator(nested.map(getAgg).toArray)
    case GroupedAggSig(k, nested) =>
      new GroupedAggregator(k, nested.map(getAgg).toArray)
    case PhysicalAggSig(NDArraySum(), NDArraySumStateSig(nda)) =>
      new NDArraySumAggregator(nda)
  }

  def apply(ir: IR, resultName: String, r: RequirednessAnalysis, isScan: Boolean = false): Aggs = {
    val ab = new BoxedArrayBuilder[(InitOp, PhysicalAggSig)]()
    val seq = new BoxedArrayBuilder[IR]()
    val let = new BoxedArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, seq, let, ref, r, isScan)
    val (initOps, pAggSigs) = ab.result().unzip
    val rt = TTuple(initOps.map(_.aggSig.resultType): _*)
    ref._typ = rt

    Aggs(postAgg, Begin(initOps), addLets(Begin(seq.result()), let.result()), pAggSigs)
  }

  private def extract(ir: IR, ab: BoxedArrayBuilder[(InitOp, PhysicalAggSig)], seqBuilder: BoxedArrayBuilder[IR], letBuilder: BoxedArrayBuilder[AggLet], result: IR, r: RequirednessAnalysis, isScan: Boolean): IR = {
    def extract(node: IR): IR = this.extract(node, ab, seqBuilder, letBuilder, result, r, isScan)

    ir match {
      case x@AggLet(name, value, body, _) =>
        letBuilder += x
        extract(body)
      case x: ApplyAggOp if !isScan =>
        val i = ab.length
        val op = x.aggSig.op
        val state = PhysicalAggSig(op, AggStateSig(op, x.initOpArgs, x.seqOpArgs, r))
        ab += InitOp(i, x.initOpArgs, state) -> state
        seqBuilder += SeqOp(i, x.seqOpArgs, state)
        GetTupleElement(result, i)
      case x: ApplyScanOp if isScan =>
        val i = ab.length
        val op = x.aggSig.op
        val state = PhysicalAggSig(op, AggStateSig(op, x.initOpArgs, x.seqOpArgs, r))
        ab += InitOp(i, x.initOpArgs, state) -> state
        seqBuilder += SeqOp(i, x.seqOpArgs, state)
        GetTupleElement(result, i)
      case AggFilter(cond, aggIR, _) =>
        val newSeq = new BoxedArrayBuilder[IR]()
        val newLet = new BoxedArrayBuilder[AggLet]()
        val transformed = this.extract(aggIR, ab, newSeq, newLet, result, r, isScan)

        seqBuilder += If(cond, addLets(Begin(newSeq.result()), newLet.result()), Begin(FastIndexedSeq[IR]()))
        transformed

      case AggExplode(array, name, aggBody, _) =>
        val newSeq = new BoxedArrayBuilder[IR]()
        val newLet = new BoxedArrayBuilder[AggLet]()
        val transformed = this.extract(aggBody, ab, newSeq, newLet, result, r, isScan)

        val (dependent, independent) = partitionDependentLets(newLet.result(), name)
        letBuilder ++= independent
        seqBuilder += StreamFor(array, name, addLets(Begin(newSeq.result()), dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        val newAggs = new BoxedArrayBuilder[(InitOp, PhysicalAggSig)]()
        val newSeq = new BoxedArrayBuilder[IR]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggIR, newAggs, newSeq, letBuilder, GetField(newRef, "value"), r, isScan)

        val i = ab.length
        val (initOps, pAggSigs) = newAggs.result().unzip

        val rt = TDict(key.typ, TTuple(initOps.map(_.aggSig.resultType): _*))
        newRef._typ = rt.elementType

        val groupState = AggStateSig.grouped(key, pAggSigs.map(_.state), r)
        val groupSig = GroupedAggSig(groupState.kt, pAggSigs.toFastSeq)
        ab += InitOp(i, FastIndexedSeq(Begin(initOps)), groupSig) -> groupSig
        seqBuilder += SeqOp(i, FastIndexedSeq(key, Begin(newSeq.result().toFastIndexedSeq)), groupSig)

        ToDict(StreamMap(ToStream(GetTupleElement(result, i)), newRef.name, MakeTuple.ordered(FastSeq(GetField(newRef, "key"), transformed))))

      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, _) =>
        val newAggs = new BoxedArrayBuilder[(InitOp, PhysicalAggSig)]()
        val newSeq = new BoxedArrayBuilder[IR]()
        val newLet = new BoxedArrayBuilder[AggLet]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggBody, newAggs, newSeq, newLet, newRef, r, isScan)

        val (dependent, independent) = partitionDependentLets(newLet.result(), elementName)
        letBuilder ++= independent

        val i = ab.length
        val (initOps, pAggSigs) = newAggs.result().unzip

        val rt = TArray(TTuple(initOps.map(_.aggSig.resultType): _*))
        newRef._typ = rt.elementType

        val checkSig = ArrayLenAggSig(knownLength.isDefined, pAggSigs)
        val eltSig = AggElementsAggSig(pAggSigs)

        val aRef = Ref(genUID(), a.typ)
        val iRef = Ref(genUID(), TInt32)

        ab += InitOp(i, knownLength.map(FastSeq(_)).getOrElse(FastSeq[IR]()) :+ Begin(initOps), checkSig) -> checkSig
        seqBuilder +=
          Let(
            aRef.name, a,
            Begin(FastIndexedSeq(
              SeqOp(i, FastIndexedSeq(ArrayLen(aRef)), checkSig),
              StreamFor(
                StreamRange(I32(0), ArrayLen(aRef), I32(1)),
                iRef.name,
                Let(
                  elementName,
                  ArrayRef(aRef, iRef),
                  addLets(SeqOp(i,
                    FastIndexedSeq(iRef, Begin(newSeq.result().toFastIndexedSeq)),
                    eltSig), dependent))))))

        val rUID = Ref(genUID(), rt)
        Let(
          rUID.name,
          GetTupleElement(result, i),
          ToArray(StreamMap(
            StreamRange(0, ArrayLen(rUID), 1),
            indexName,
            Let(
              newRef.name,
              ArrayRef(rUID, Ref(indexName, TInt32)),
              transformed))))

      case x: StreamAgg =>
        assert(!ContainsScan(x))
        x
      case x: StreamAggScan =>
        assert(!ContainsAgg(x))
        x
      case _ => MapIR(extract)(ir)
    }
  }
}
