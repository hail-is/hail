package is.hail.expr.ir.agg

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue}
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.BufferSpec
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.utils._

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

class UnsupportedExtraction(msg: String) extends Exception(msg)

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[PhysicalAggSignature]) {
  val typ: PTuple = PTuple(aggs.map(Extract.getPType): _*)
  val nAggs: Int = aggs.length

  def isCommutative: Boolean = {
    def aggCommutes(agg: PhysicalAggSignature): Boolean = agg.nested.forall(_.forall(aggCommutes)) && AggIsCommutative(agg.op)

    aggs.forall(aggCommutes)
  }

  def shouldTreeAggregate: Boolean = {
    def containsBigAggregator(agg: PhysicalAggSignature): Boolean = agg.nested.exists(_.exists(containsBigAggregator)) || (agg.op match {
      case AggElements() => true
      case AggElementsLengthCheck() => true
      case Downsample() => true
      case _ => false
    })

    aggs.exists(containsBigAggregator)
  }

  def deserializeSet(i: Int, i2: Int, spec: BufferSpec): IR =
    DeserializeAggs(i * nAggs, i2, spec, aggs)

  def serializeSet(i: Int, i2: Int, spec: BufferSpec): IR =
    SerializeAggs(i * nAggs, i2, spec, aggs)

  def eltOp(ctx: ExecuteContext, optimize: Boolean = true): IR = if (optimize)
    Optimize(seqPerElt, true, "eltOp", ctx)
  else
    seqPerElt

  def deserialize(ctx: ExecuteContext, spec: BufferSpec): ((Region, Array[Byte]) => Long) = {
    val (_, f) = ir.CompileWithAggregators2[Unit](ctx,
      aggs, ir.DeserializeAggs(0, 0, spec, aggs))

    { (aggRegion: Region, bytes: Array[Byte]) =>
      val f2 = f(0, aggRegion);
      f2.newAggState(aggRegion)
      f2.setSerializedAgg(0, bytes)
      f2(aggRegion)
      f2.getAggOffset()
    }
  }

  def serialize(ctx: ExecuteContext, spec: BufferSpec): (Region, Long) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators2[Unit](ctx,
      aggs, ir.SerializeAggs(0, 0, spec, aggs))

    { (aggRegion: Region, off: Long) =>
      val f2 = f(0, aggRegion);
      f2.setAggState(aggRegion, off)
      f2(aggRegion)
      f2.getSerializedAgg(0)
    }
  }

  def combOpF(ctx: ExecuteContext, spec: BufferSpec): (Array[Byte], Array[Byte]) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators2[Unit](ctx,
      aggs ++ aggs,
      Begin(
        deserializeSet(0, 0, spec) +:
          deserializeSet(1, 1, spec) +:
          Array.tabulate(nAggs)(i => CombOp2(i, nAggs + i, aggs(i))) :+
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

  def results: IR = ResultOp2(0, aggs)
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

  def compatible(sig1: PhysicalAggSignature, sig2: PhysicalAggSignature): Boolean = (sig1.op, sig2.op) match {
    case (AggElements(), AggElements()) |
         (AggElementsLengthCheck(), AggElements()) |
         (AggElements(), AggElementsLengthCheck()) |
         (AggElementsLengthCheck(), AggElementsLengthCheck()) =>
      sig1.nested.get.zip(sig2.nested.get).forall { case (a1, a2) => compatible(a1, a2) }
    case _ => sig1 == sig2
  }

  def getResultType(aggSig: AggSignature): Type = aggSig match {
    case AggSignature(Sum(), _, Seq(t), _) => t
    case AggSignature(Product(), _, Seq(t), _) => t
    case AggSignature(Min(), _, Seq(t), _) => t
    case AggSignature(Max(), _, Seq(t), _) => t
    case AggSignature(Count(), _, _, _) => TInt64()
    case AggSignature(Take(), _, Seq(t), _) => TArray(t)
    case AggSignature(CallStats(), _, _, _) => CallStatsState.resultType.virtualType
    case AggSignature(TakeBy(), _, Seq(value, key), _) => TArray(value)
    case AggSignature(PrevNonnull(), _, Seq(t), _) => t
    case AggSignature(CollectAsSet(), _, Seq(t), _) => TSet(t)
    case AggSignature(Collect(), _, Seq(t), _) => TArray(t)
    case AggSignature(LinearRegression(), _, _, _) =>
      LinearRegressionAggregator.resultType.virtualType
    case AggSignature(ApproxCDF(), _, _, _) => QuantilesAggregator.resultType.virtualType
    case AggSignature(Downsample(), _, Seq(_, _, label), _) => DownsampleAggregator.resultType
    case _ => throw new UnsupportedExtraction(aggSig.toString)  }

  def getAgg(aggSig: PhysicalAggSignature): StagedAggregator = aggSig match {
    case PhysicalAggSignature(Sum(), _, Seq(t), _) =>
      new SumAggregator(t)
    case PhysicalAggSignature(Product(), _, Seq(t), _) =>
      new ProductAggregator(t)
    case PhysicalAggSignature(Min(), _, Seq(t), _) =>
      new MinAggregator(t)
    case PhysicalAggSignature(Max(), _, Seq(t), _) =>
      new MaxAggregator(t)
    case PhysicalAggSignature(Count(), _, _, _) =>
      CountAggregator
    case PhysicalAggSignature(Take(), _, Seq(t), _) => new TakeAggregator(t)
    case PhysicalAggSignature(CallStats(), _, Seq(tCall: PCall), _) => new CallStatsAggregator(tCall)
    case PhysicalAggSignature(TakeBy(), _, Seq(value, key), _) => new TakeByAggregator(value, key)
    case PhysicalAggSignature(AggElementsLengthCheck(), initOpArgs, _, Some(nestedAggs)) =>
      val knownLength = initOpArgs.length == 2
      new ArrayElementLengthCheckAggregator(nestedAggs.map(getAgg).toArray, knownLength)
    case PhysicalAggSignature(AggElements(), _, _, Some(nestedAggs)) =>
      new ArrayElementwiseOpAggregator(nestedAggs.map(getAgg).toArray)
    case PhysicalAggSignature(PrevNonnull(), _, Seq(t), _) =>
      new PrevNonNullAggregator(t)
    case PhysicalAggSignature(Group(), _, Seq(kt, PVoid), Some(nestedAggs)) =>
      new GroupedAggregator(PType.canonical(kt), nestedAggs.map(getAgg).toArray)
    case PhysicalAggSignature(CollectAsSet(), _, Seq(t), _) =>
      new CollectAsSetAggregator(PType.canonical(t))
    case PhysicalAggSignature(Collect(), _, Seq(t), _) =>
      new CollectAggregator(t)
    case PhysicalAggSignature(LinearRegression(), _, _, _) =>
      LinearRegressionAggregator
    case PhysicalAggSignature(ApproxCDF(), _, _, _) => new ApproxCDFAggregator
    case PhysicalAggSignature(Downsample(), _, Seq(_, _, label), _) => new DownsampleAggregator(label.asInstanceOf[PArray])
    case _ => throw new UnsupportedExtraction(aggSig.toString)
  }

  def getPType(aggSig: PhysicalAggSignature): PType = getAgg(aggSig).resultType

  def apply(ir: IR, resultName: String): Aggs = {
    val ab = new ArrayBuilder[InitOp2]()
    val seq = new ArrayBuilder[IR]()
    val let = new ArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, seq, let, ref)
    val initOps = ab.result()
    val aggs = initOps.map(_.aggSig)
    val rt = TTuple(aggs.map(_.returnType.virtualType): _*)
    ref._typ = rt

    Aggs(postAgg, Begin(initOps), addLets(Begin(seq.result()), let.result()), aggs)
  }

  private def extract(ir: IR, ab: ArrayBuilder[InitOp2], seqBuilder: ArrayBuilder[IR], letBuilder: ArrayBuilder[AggLet], result: IR): IR = {
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
        val psig = x.aggSig.toPhysical(x.initOpArgs.map(_.pType), x.seqOpArgs.map(_.pType))
        ab += InitOp2(i, x.initOpArgs, psig)
        seqBuilder += SeqOp2(i, x.seqOpArgs, psig)
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
        seqBuilder += ArrayFor(array, name, addLets(Begin(newSeq.result()), dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        val newAggs = new ArrayBuilder[InitOp2]()
        val newSeq = new ArrayBuilder[IR]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggIR, newAggs, newSeq, letBuilder, GetField(newRef, "value"))

        val i = ab.length
        val initOps = newAggs.result()
        val aggs = initOps.map(_.aggSig)

        val rt = TDict(key.typ, TTuple(aggs.map(_.returnType.virtualType): _*))
        newRef._typ = -rt.elementType

        val aggSig = PhysicalAggSignature(Group(), Seq(PVoid), FastSeq(key.pType, PVoid), Some(aggs))
        ab += InitOp2(i, FastIndexedSeq(Begin(initOps)), aggSig)
        seqBuilder += SeqOp2(i, FastIndexedSeq(key, Begin(newSeq.result().toFastIndexedSeq)), aggSig)

        ToDict(ArrayMap(ToArray(GetTupleElement(result, i)), newRef.name, MakeTuple.ordered(FastSeq(GetField(newRef, "key"), transformed))))


      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, _) =>
        val newAggs = new ArrayBuilder[InitOp2]()
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggBody, newAggs, newSeq, newLet, newRef)

        val (dependent, independent) = partitionDependentLets(newLet.result(), elementName)
        letBuilder ++= independent

        val i = ab.length
        val initOps = newAggs.result()
        val aggs = initOps.map(_.aggSig)

        val rt = TArray(TTuple(aggs.map(_.returnType.virtualType): _*))
        newRef._typ = -rt.elementType

        val aggSigCheck = PhysicalAggSignature(
          AggElementsLengthCheck(),
          knownLength.map(l => FastSeq(l.pType)).getOrElse(FastSeq()) :+ PVoid,
          FastSeq(PInt32()), Some(aggs))
        val aggSig = PhysicalAggSignature(AggElements(), FastSeq[PType](), FastSeq(PInt32(), PVoid), Some(aggs))

        val aRef = Ref(genUID(), a.typ)
        val iRef = Ref(genUID(), TInt32())

        ab += InitOp2(i, knownLength.map(FastSeq(_)).getOrElse(FastSeq[IR]()) :+ Begin(initOps), aggSigCheck)
        seqBuilder +=
          Let(
            aRef.name, a,
            Begin(FastIndexedSeq(
              SeqOp2(i, FastIndexedSeq(ArrayLen(aRef)), aggSigCheck),
              ArrayFor(
                ArrayRange(I32(0), ArrayLen(aRef), I32(1)),
                iRef.name,
                Let(
                  elementName,
                  ArrayRef(aRef, iRef),
                  addLets(SeqOp2(i,
                    FastIndexedSeq(iRef, Begin(newSeq.result().toFastIndexedSeq)),
                    aggSig), dependent))))))

        val rUID = Ref(genUID(), rt)
        Let(
          rUID.name,
          GetTupleElement(result, i),
          ArrayMap(
            ArrayRange(0, ArrayLen(rUID), 1),
            indexName,
            Let(
              newRef.name,
              ArrayRef(rUID, Ref(indexName, TInt32())),
              transformed)))

      case x: ArrayAgg =>
        assert(!ContainsScan(x))
        x
      case x: ArrayAggScan =>
        assert(!ContainsAgg(x))
        x
      case _ => MapIR(extract)(ir)
    }
  }
}
