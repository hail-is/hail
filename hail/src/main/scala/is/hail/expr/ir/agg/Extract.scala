package is.hail.expr.ir.agg

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue}
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.BufferSpec
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.utils._

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

class UnsupportedExtraction(msg: String) extends Exception(msg)

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[AggStateSignature]) {
  val nAggs: Int = aggs.length

  def isCommutative: Boolean = {
    def aggCommutes(agg: AggStateSignature): Boolean = agg.m.keysIterator.forall(AggIsCommutative(_)) && agg.nested.forall(_.forall(aggCommutes))

    aggs.forall(aggCommutes)
  }

  def shouldTreeAggregate: Boolean = {
    def containsBigAggregator(agg: AggStateSignature): Boolean = (agg.m.keysIterator.exists {
      case AggElements() => true
      case AggElementsLengthCheck() => true
      case Downsample() => true
      case _ => false
    }) || agg.nested.exists(_.exists(containsBigAggregator))

    aggs.exists(containsBigAggregator)
  }

  def deserializeSet(i: Int, i2: Int, spec: BufferSpec): IR =
    DeserializeAggs(i * nAggs, i2, spec, aggs)

  def serializeSet(i: Int, i2: Int, spec: BufferSpec): IR =
    SerializeAggs(i * nAggs, i2, spec, aggs)

  def eltOp(ctx: ExecuteContext): IR = seqPerElt

  def deserialize(ctx: ExecuteContext, spec: BufferSpec, physicalAggs: Array[AggStatePhysicalSignature]): ((Region, Array[Byte]) => Long) = {
    val (_, f) = ir.CompileWithAggregators2[Unit](ctx,
      physicalAggs, ir.DeserializeAggs(0, 0, spec, aggs))

    { (aggRegion: Region, bytes: Array[Byte]) =>
      val f2 = f(0, aggRegion);
      f2.newAggState(aggRegion)
      f2.setSerializedAgg(0, bytes)
      f2(aggRegion)
      f2.getAggOffset()
    }
  }

  def serialize(ctx: ExecuteContext, spec: BufferSpec, physicalAggs: Array[AggStatePhysicalSignature]): (Region, Long) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators2[Unit](ctx,
      physicalAggs, ir.SerializeAggs(0, 0, spec, aggs))

    { (aggRegion: Region, off: Long) =>
      val f2 = f(0, aggRegion);
      f2.setAggState(aggRegion, off)
      f2(aggRegion)
      f2.getSerializedAgg(0)
    }
  }

  def combOpF(ctx: ExecuteContext, spec: BufferSpec, physicalAggs: Array[AggStatePhysicalSignature]): (Array[Byte], Array[Byte]) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators2[Unit](ctx,
      physicalAggs ++ physicalAggs,
      Begin(
        deserializeSet(0, 0, spec) +:
          deserializeSet(1, 1, spec) +:
          Array.tabulate(nAggs)(i => CombOp(i, nAggs + i, aggs(i))) :+
          serializeSet(0, 0, spec)))

    { (c1: Array[Byte], c2: Array[Byte]) =>
      Region.smallScoped { aggRegion =>
        val comb = f(0, aggRegion)
        comb.newAggState(aggRegion)
        comb.setSerializedAgg(0, c1)
        comb.setSerializedAgg(1, c2)
        comb(aggRegion)
        comb.getSerializedAgg(0)
      }
    }
  }

  def results: IR = ResultOp(0, aggs)

  def getPhysicalAggs(ctx: ExecuteContext, initBindings: Env[PType], seqBindings: Env[PType]): Array[AggStatePhysicalSignature] = {
    val initsAB = InferPType.newBuilder[InitOp](aggs.length)
    val seqsAB = InferPType.newBuilder[SeqOp](aggs.length)
    val init2 = LoweringPipeline.compileLowerer.apply(ctx, init, false).asInstanceOf[IR].noSharing
    val seq2 = LoweringPipeline.compileLowerer.apply(ctx, seqPerElt, false).asInstanceOf[IR].noSharing
    InferPType(init2, initBindings, null, inits = initsAB, null)
    InferPType(seq2, seqBindings, null, null, seqs = seqsAB)

    val pSigs = aggs.indices.map { i => InferPType.computePhysicalAgg(aggs(i), initsAB(i), seqsAB(i)) }.toArray

    if (init2 eq init)
      InferPType.clearPTypes(init2)
    if (seq2 eq seqPerElt)
      InferPType.clearPTypes(seq2)

    pSigs
  }
}

object Extract {
  def liftScan(ir: IR): IR = ir match {
    case ApplyScanOp(init, seq, sig) => ApplyAggOp(init, seq, sig)
    case x => MapIR(liftScan)(x)
  }

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

  def compatible(sig1: AggStatePhysicalSignature, sig2: AggStatePhysicalSignature): Boolean = sig1.default == sig2.default

  def getResultType(aggSig: AggStateSignature): Type = aggSig.defaultSignature match {
    case AggSignature(Sum(), _, Seq(t)) => t
    case AggSignature(Product(), _, Seq(t)) => t
    case AggSignature(Min(), _, Seq(t)) => t
    case AggSignature(Max(), _, Seq(t)) => t
    case AggSignature(Count(), _, _) => TInt64
    case AggSignature(Take(), _, Seq(t)) => TArray(t)
    case AggSignature(CallStats(), _, _) => CallStatsState.resultType.virtualType
    case AggSignature(TakeBy(), _, Seq(value, key)) => TArray(value)
    case AggSignature(PrevNonnull(), _, Seq(t)) => t
    case AggSignature(CollectAsSet(), _, Seq(t)) => TSet(t)
    case AggSignature(Collect(), _, Seq(t)) => TArray(t)
    case AggSignature(LinearRegression(), _, _) =>
      LinearRegressionAggregator.resultType.virtualType
    case AggSignature(ApproxCDF(), _, _) => QuantilesAggregator.resultType.virtualType
    case AggSignature(Downsample(), _, Seq(_, _, label)) => DownsampleAggregator.resultType
    case AggSignature(AggElementsLengthCheck(), _, _) => TArray(TTuple(aggSig.nested.get.map(getResultType): _*))
    case AggSignature(Group(), _, Seq(k, _)) =>  TDict(k, TTuple(aggSig.nested.get.map(getResultType): _*))
    case _ => throw new UnsupportedExtraction(aggSig.toString)  }

  def getAgg(aggSig: AggStatePhysicalSignature, op: AggOp): StagedAggregator = aggSig.lookup(op) match {
    case PhysicalAggSignature(Sum(), _, Seq(t)) =>
      new SumAggregator(t)
    case PhysicalAggSignature(Product(), _, Seq(t)) =>
      new ProductAggregator(t)
    case PhysicalAggSignature(Min(), _, Seq(t)) =>
      new MinAggregator(t)
    case PhysicalAggSignature(Max(), _, Seq(t)) =>
      new MaxAggregator(t)
    case PhysicalAggSignature(Count(), _, _) =>
      CountAggregator
    case PhysicalAggSignature(Take(), _, Seq(t)) => new TakeAggregator(t)
    case PhysicalAggSignature(CallStats(), _, Seq(tCall: PCall)) => new CallStatsAggregator(tCall)
    case PhysicalAggSignature(TakeBy(), _, Seq(value, key)) => new TakeByAggregator(value, key)
    case PhysicalAggSignature(AggElementsLengthCheck(), initOpArgs, _) =>
      val knownLength = initOpArgs.length == 2
      new ArrayElementLengthCheckAggregator(aggSig.nested.get.map(a => getAgg(a, a.default)).toArray, knownLength)
    case PhysicalAggSignature(AggElements(), _, _) =>
      new ArrayElementwiseOpAggregator(aggSig.nested.get.map(a => getAgg(a, a.default)).toArray)
    case PhysicalAggSignature(PrevNonnull(), _, Seq(t)) =>
      new PrevNonNullAggregator(t)
    case PhysicalAggSignature(Group(), _, Seq(kt, PVoid)) =>
      new GroupedAggregator(PType.canonical(kt), aggSig.nested.get.map(a => getAgg(a, a.default)).toArray)
    case PhysicalAggSignature(CollectAsSet(), _, Seq(t)) =>
      new CollectAsSetAggregator(PType.canonical(t))
    case PhysicalAggSignature(Collect(), _, Seq(t)) =>
      new CollectAggregator(t)
    case PhysicalAggSignature(LinearRegression(), _, _) =>
      LinearRegressionAggregator
    case PhysicalAggSignature(ApproxCDF(), _, _) => new ApproxCDFAggregator
    case PhysicalAggSignature(Downsample(), _, Seq(_, _, label)) => new DownsampleAggregator(label.asInstanceOf[PArray])
    case _ => throw new UnsupportedExtraction(aggSig.toString)
  }

  def getPType(aggSig: AggStatePhysicalSignature): PType = getAgg(aggSig, aggSig.default).resultType

  def apply(ir: IR, resultName: String): Aggs = {
    val ab = new ArrayBuilder[InitOp]()
    val seq = new ArrayBuilder[IR]()
    val let = new ArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, seq, let, ref)
    val initOps = ab.result()
    val rt = TTuple(initOps.map(_.aggSig.resultType): _*)
    ref._typ = rt

    Aggs(postAgg, Begin(initOps), addLets(Begin(seq.result()), let.result()), initOps.map(_.aggSig))
  }

  private def extract(ir: IR, ab: ArrayBuilder[InitOp], seqBuilder: ArrayBuilder[IR], letBuilder: ArrayBuilder[AggLet], result: IR): IR = {
    def extract(node: IR): IR = this.extract(node, ab, seqBuilder, letBuilder, result)

    ir match {
      case Ref(name, typ) =>
        assert(typ.isRealizable)
        ir
      case x@AggLet(name, value, body, _) =>
        letBuilder += x
        extract(body)
      case x: ApplyAggOp =>
        val i = ab.length
        val sig = x.aggSig
        val state = AggStateSignature(sig)
        val op = sig.op
        ab += ((InitOp(i, x.initOpArgs, state, op)))
        seqBuilder += SeqOp(i, x.seqOpArgs, state, op)
        GetTupleElement(result, i)
      case AggFilter(cond, aggIR, _) =>
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val transformed = this.extract(aggIR, ab, newSeq, newLet, result)

        seqBuilder += If(cond, addLets(Begin(newSeq.result()), newLet.result()), Begin(FastIndexedSeq[IR]()))
        transformed

      case AggExplode(array, name, aggBody, _) =>
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val transformed = this.extract(aggBody, ab, newSeq, newLet, result)

        val (dependent, independent) = partitionDependentLets(newLet.result(), name)
        letBuilder ++= independent
        seqBuilder += StreamFor(array, name, addLets(Begin(newSeq.result()), dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        val newAggs = new ArrayBuilder[InitOp]()
        val newSeq = new ArrayBuilder[IR]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggIR, newAggs, newSeq, letBuilder, GetField(newRef, "value"))

        val i = ab.length
        val initOps = newAggs.result()

        val rt = TDict(key.typ, TTuple(initOps.map(_.aggSig.resultType): _*))
        newRef._typ = rt.elementType

        // the void-typed init and seq args are side-effecting agg IRs (InitOp and SeqOp nodes for sub-aggs)
        val groupSig = AggSignature(Group(), Seq(TVoid), FastSeq(key.typ, TVoid))
        val aggSig = AggStateSignature(Map(Group() -> groupSig), Group(), Some(initOps.map(_.aggSig)))
        ab += InitOp(i, FastIndexedSeq(Begin(initOps)), aggSig, Group())
        seqBuilder += SeqOp(i, FastIndexedSeq(key, Begin(newSeq.result().toFastIndexedSeq)), aggSig, Group())

        ToDict(StreamMap(ToStream(GetTupleElement(result, i)), newRef.name, MakeTuple.ordered(FastSeq(GetField(newRef, "key"), transformed))))


      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, _) =>
        val newAggs = new ArrayBuilder[InitOp]()
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggBody, newAggs, newSeq, newLet, newRef)

        val (dependent, independent) = partitionDependentLets(newLet.result(), elementName)
        letBuilder ++= independent

        val i = ab.length
        val initOps = newAggs.result()

        val rt = TArray(TTuple(initOps.map(_.aggSig.resultType): _*))
        newRef._typ = rt.elementType

        // the void-typed init and seq args are side-effecting agg IRs (InitOp and SeqOp nodes for sub-aggs)
        val aggSigCheck = AggSignature(
          AggElementsLengthCheck(),
          knownLength.map(l => FastSeq(l.typ)).getOrElse(FastSeq()) :+ TVoid,
          FastSeq(TInt32))
        val aggSig = AggSignature(AggElements(), FastSeq(), FastSeq(TInt32, TVoid))
        val state = AggStateSignature(Map(AggElementsLengthCheck() -> aggSigCheck, AggElements() -> aggSig),
          AggElementsLengthCheck(), Some(initOps.map(_.aggSig)))

        val aRef = Ref(genUID(), a.typ)
        val iRef = Ref(genUID(), TInt32)

        ab += InitOp(i, knownLength.map(FastSeq(_)).getOrElse(FastSeq[IR]()) :+ Begin(initOps), state, AggElementsLengthCheck())
        seqBuilder +=
          Let(
            aRef.name, a,
            Begin(FastIndexedSeq(
              SeqOp(i, FastIndexedSeq(ArrayLen(aRef)), state, AggElementsLengthCheck()),
              StreamFor(
                StreamRange(I32(0), ArrayLen(aRef), I32(1)),
                iRef.name,
                Let(
                  elementName,
                  ArrayRef(aRef, iRef),
                  addLets(SeqOp(i,
                    FastIndexedSeq(iRef, Begin(newSeq.result().toFastIndexedSeq)),
                    state, AggElements()), dependent))))))

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
