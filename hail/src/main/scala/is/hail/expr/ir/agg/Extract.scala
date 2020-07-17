package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionValue}
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.io.BufferSpec
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._

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
    val seqPTypes = seqOpArgs.map { case (a, r) => if (a.typ == TVoid) PVoid else r.canonicalPType(a.typ) }
    op match {
      case Sum() | Product() => TypedStateSig(seqPTypes.head.setRequired(true))
      case Min() | Max()  => TypedStateSig(seqPTypes.head.setRequired(false))
      case Count() => TypedStateSig(PInt64(true))
      case Take() => TakeStateSig(seqPTypes.head)
      case TakeBy(reverse) =>
        val Seq(vt, kt) = seqPTypes
        TakeByStateSig(vt, kt, reverse)
      case CallStats() => CallStatsStateSig()
      case PrevNonnull() => TypedStateSig(seqPTypes.head.setRequired(false))
      case CollectAsSet() => CollectAsSetStateSig(seqPTypes.head)
      case Collect() => CollectStateSig(seqPTypes.head)
      case LinearRegression() => TypedStateSig(LinearRegressionAggregator.stateType)
      case ApproxCDF() => ApproxCDFStateSig()
      case Downsample() =>
        val Seq(_, _, labelType: PArray) = seqPTypes
        DownsampleStateSig(labelType)
      case ImputeType() => ImputeTypeStateSig()
      case _ => throw new UnsupportedExtraction(op.toString)
    }
  }
  def grouped(k: IR, aggs: Seq[AggStateSig], r: RequirednessAnalysis): GroupedStateSig =
    GroupedStateSig(r(k).canonicalPType(k.typ), aggs)
  def arrayelements(aggs: Seq[AggStateSig]): ArrayAggStateSig =
    ArrayAggStateSig(aggs)

  def getState(sig: AggStateSig, cb: EmitClassBuilder[_]): AggregatorState = sig match {
    case TypedStateSig(pt) if pt.isPrimitive => new PrimitiveRVAState(Array(pt), cb)
    case TypedStateSig(pt) => new TypedRegionBackedAggState(pt, cb)
    case DownsampleStateSig(labelType) => new DownsampleState(cb, labelType)
    case TakeStateSig(pt) => new TakeRVAS(pt, PCanonicalArray(pt, required = true), cb)
    case TakeByStateSig(vt, kt, so) => new TakeByRVAS(vt, kt, PCanonicalArray(vt, required = true), cb, so)
    case CollectStateSig(pt) => new CollectAggState(pt, cb)
    case CollectAsSetStateSig(pt) => new AppendOnlySetState(cb, pt)
    case CallStatsStateSig() => new CallStatsState(cb)
    case ApproxCDFStateSig() => new ApproxCDFState(cb)
    case ImputeTypeStateSig() => new ImputeTypeState(cb)
    case ArrayAggStateSig(nested) => new ArrayElementState(cb, StateTuple(nested.map(sig => AggStateSig.getState(sig, cb)).toArray))
    case GroupedStateSig(kt, nested) => new DictState(cb, kt, StateTuple(nested.map(sig => AggStateSig.getState(sig, cb)).toArray))
  }
}

sealed abstract class AggStateSig(val t: Seq[PType], val n: Option[Seq[AggStateSig]])
case class TypedStateSig(pt: PType) extends AggStateSig(Array(pt), None)
case class DownsampleStateSig(labelType: PArray) extends AggStateSig(Array(labelType), None)
case class TakeStateSig(pt: PType) extends AggStateSig(Array(pt), None)
case class TakeByStateSig(vt: PType, kt: PType, so: SortOrder) extends AggStateSig(Array(vt, kt), None)
case class CollectStateSig(pt: PType) extends AggStateSig(Array(pt), None)
case class CollectAsSetStateSig(pt: PType) extends AggStateSig(Array(pt), None)
case class CallStatsStateSig() extends AggStateSig(Array[PType](), None)
case class ImputeTypeStateSig() extends AggStateSig(Array[PType](), None)
case class ArrayAggStateSig(nested: Seq[AggStateSig]) extends AggStateSig(Array[PType](), Some(nested))
case class GroupedStateSig(kt: PType, nested: Seq[AggStateSig]) extends AggStateSig(Array(kt), Some(nested))
case class ApproxCDFStateSig() extends AggStateSig(Array[PType](), None)

object PhysicalAggSig {
  def apply(op: AggOp, state: AggStateSig): PhysicalAggSig = BasicPhysicalAggSig(op, state)
  def unapply(v: PhysicalAggSig): Option[(AggOp, AggStateSig)] = if (v.nestedOps.isEmpty) Some(v.op -> v.state) else None
}

class PhysicalAggSig(val op: AggOp, val state: AggStateSig, val nestedOps: Array[AggOp]) {
  val allOps: Array[AggOp] = nestedOps :+ op
  def initOpTypes: IndexedSeq[Type] = Extract.getAgg(this).initOpTypes.map(_.virtualType).toFastIndexedSeq
  def seqOpTypes: IndexedSeq[Type] = Extract.getAgg(this).seqOpTypes.map(_.virtualType).toFastIndexedSeq
  def pResultType: PType = Extract.getAgg(this).resultType
  def resultType: Type = pResultType.virtualType
}

case class BasicPhysicalAggSig(override val op: AggOp, override val state: AggStateSig) extends PhysicalAggSig(op, state, Array())

case class GroupedAggSig(kt: PType, nested: Seq[PhysicalAggSig]) extends
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

    { (aggRegion: Region, bytes: Array[Byte]) =>
      val f2 = f(0, aggRegion)
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

    { (aggRegion: Region, off: Long) =>
      val f2 = f(0, aggRegion)
      f2.setAggState(aggRegion, off)
      f2(aggRegion)
      f2.getSerializedAgg(0)
    }
  }

  def combOpFSerialized(ctx: ExecuteContext, spec: BufferSpec): (Array[Byte], Array[Byte]) => Array[Byte] = {
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

    { (bytes1: Array[Byte], bytes2: Array[Byte]) =>
      Region.smallScoped { r =>
        val f2 = f(0, r)
        f2.newAggState(r)
        f2.setSerializedAgg(0, bytes1)
        f2.setSerializedAgg(1, bytes2)
        f2(r)
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

    { (l: RegionValue, r: RegionValue) =>
      val comb = f(0, l.region)
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

    val dep = new ArrayBuilder[AggLet]
    val indep = new ArrayBuilder[AggLet]

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
    case AggSignature(ImputeType(), _, _) => ImputeTypeState.resultType.virtualType
    case AggSignature(LinearRegression(), _, _) =>
      LinearRegressionAggregator.resultType.virtualType
    case AggSignature(ApproxCDF(), _, _) => QuantilesAggregator.resultType.virtualType
    case AggSignature(Downsample(), _, Seq(_, _, label)) => DownsampleAggregator.resultType
    case _ => throw new UnsupportedExtraction(aggSig.toString)  }

  def getAgg(sig: PhysicalAggSig): StagedAggregator = sig match {
    case PhysicalAggSig(Sum(), TypedStateSig(t)) => new SumAggregator(t)
    case PhysicalAggSig(Product(), TypedStateSig(t)) => new ProductAggregator(t)
    case PhysicalAggSig(Min(), TypedStateSig(t)) => new MinAggregator(t)
    case PhysicalAggSig(Max(), TypedStateSig(t)) => new MaxAggregator(t)
    case PhysicalAggSig(PrevNonnull(), TypedStateSig(t)) => new PrevNonNullAggregator(t)
    case PhysicalAggSig(Count(), TypedStateSig(_)) => CountAggregator
    case PhysicalAggSig(Take(), TakeStateSig(t)) => new TakeAggregator(t)
    case PhysicalAggSig(TakeBy(_), TakeByStateSig(v, k, _)) => new TakeByAggregator(v, k)
    case PhysicalAggSig(CallStats(), CallStatsStateSig()) => new CallStatsAggregator(PCanonicalCall()) // FIXME CallStatsAggregator shouldn't take type
    case PhysicalAggSig(Collect(), CollectStateSig(t)) => new CollectAggregator(t)
    case PhysicalAggSig(CollectAsSet(), CollectAsSetStateSig(t)) => new CollectAsSetAggregator(t)
    case PhysicalAggSig(LinearRegression(), TypedStateSig(_)) => new LinearRegressionAggregator(PFloat64(), PCanonicalArray(PFloat64())) // FIXME LinRegAggregator shouldn't take type
    case PhysicalAggSig(ApproxCDF(), ApproxCDFStateSig()) => new ApproxCDFAggregator
    case PhysicalAggSig(Downsample(), DownsampleStateSig(labelType)) => new DownsampleAggregator(labelType)
    case PhysicalAggSig(ImputeType(), ImputeTypeStateSig()) => new ImputeTypeAggregator(PCanonicalString())
    case ArrayLenAggSig(knownLength, nested) => //FIXME nested things shouldn't need to know state
      new ArrayElementLengthCheckAggregator(nested.map(getAgg).toArray, knownLength)
    case AggElementsAggSig(nested) =>
      new ArrayElementwiseOpAggregator(nested.map(getAgg).toArray)
    case GroupedAggSig(k, nested) =>
      new GroupedAggregator(k, nested.map(getAgg).toArray)
  }

  def apply(ir: IR, resultName: String, r: RequirednessAnalysis, isScan: Boolean = false): Aggs = {
    val ab = new ArrayBuilder[(InitOp, PhysicalAggSig)]()
    val seq = new ArrayBuilder[IR]()
    val let = new ArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, seq, let, ref, r, isScan)
    val (initOps, pAggSigs) = ab.result().unzip
    val rt = TTuple(initOps.map(_.aggSig.resultType): _*)
    ref._typ = rt

    Aggs(postAgg, Begin(initOps), addLets(Begin(seq.result()), let.result()), pAggSigs)
  }

  private def extract(ir: IR, ab: ArrayBuilder[(InitOp, PhysicalAggSig)], seqBuilder: ArrayBuilder[IR], letBuilder: ArrayBuilder[AggLet], result: IR, r: RequirednessAnalysis, isScan: Boolean): IR = {
    def extract(node: IR): IR = this.extract(node, ab, seqBuilder, letBuilder, result, r, isScan)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
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
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val transformed = this.extract(aggIR, ab, newSeq, newLet, result, r, isScan)

        seqBuilder += If(cond, addLets(Begin(newSeq.result()), newLet.result()), Begin(FastIndexedSeq[IR]()))
        transformed

      case AggExplode(array, name, aggBody, _) =>
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val transformed = this.extract(aggBody, ab, newSeq, newLet, result, r, isScan)

        val (dependent, independent) = partitionDependentLets(newLet.result(), name)
        letBuilder ++= independent
        seqBuilder += StreamFor(array, name, addLets(Begin(newSeq.result()), dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        val newAggs = new ArrayBuilder[(InitOp, PhysicalAggSig)]()
        val newSeq = new ArrayBuilder[IR]()
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
        val newAggs = new ArrayBuilder[(InitOp, PhysicalAggSig)]()
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
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
