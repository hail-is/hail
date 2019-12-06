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

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[AggSignature2]) {
  val typ: PTuple = PTuple(aggs.map(Extract.getPType): _*)
  val nAggs: Int = aggs.length

  def isCommutative: Boolean = {
    def aggCommutes(agg: AggSignature2): Boolean = agg.nested.forall(_.forall(aggCommutes)) && AggIsCommutative(agg.op)
    aggs.forall(aggCommutes)
  }

  def shouldTreeAggregate: Boolean = {
    def containsBigAggregator(agg: AggSignature2): Boolean = agg.nested.exists(_.exists(containsBigAggregator)) || (agg.op match {
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
    lets.foldRight[IR](ir) { case (al, comb) => Let(al.name, al.value, comb)}
  }

  def compatible(sig1: AggSignature2, sig2: AggSignature2): Boolean = (sig1.op, sig2.op) match {
    case (AggElements(), AggElements()) |
         (AggElementsLengthCheck(), AggElements()) |
         (AggElements(), AggElementsLengthCheck()) |
         (AggElementsLengthCheck(), AggElementsLengthCheck()) =>
      sig1.nested.get.zip(sig2.nested.get).forall { case (a1, a2) => compatible(a1, a2) }
    case _ => sig1 == sig2
  }

  def getAgg(aggSig: AggSignature2): StagedAggregator = aggSig match {
    case AggSignature2(Sum(), _, Seq(t), _) =>
      new SumAggregator(t.physicalType)
    case AggSignature2(Product(), _, Seq(t), _) =>
      new ProductAggregator(t.physicalType)
    case AggSignature2(Min(), _, Seq(t), _) =>
      new MinAggregator(t.physicalType)
    case AggSignature2(Max(), _, Seq(t), _) =>
      new MaxAggregator(t.physicalType)
    case AggSignature2(Count(), _, _, _) =>
      CountAggregator
    case AggSignature2(Take(), _, Seq(t), _) => new TakeAggregator(t.physicalType)
    case AggSignature2(CallStats(), _, Seq(tCall: TCall), _) => new CallStatsAggregator(tCall.physicalType)
    case AggSignature2(TakeBy(), _, Seq(value, key), _) => new TakeByAggregator(value.physicalType, key.physicalType)
    case AggSignature2(AggElementsLengthCheck(), initOpArgs, _, Some(nestedAggs)) =>
      val knownLength = initOpArgs.length == 2
      new ArrayElementLengthCheckAggregator(nestedAggs.map(getAgg).toArray, knownLength)
    case AggSignature2(AggElements(), _, _, Some(nestedAggs)) =>
      new ArrayElementwiseOpAggregator(nestedAggs.map(getAgg).toArray)
    case AggSignature2(PrevNonnull(), _, Seq(t), _) =>
      new PrevNonNullAggregator(t.physicalType)
    case AggSignature2(Group(), _, Seq(kt, TVoid), Some(nestedAggs)) =>
      new GroupedAggregator(PType.canonical(kt), nestedAggs.map(getAgg).toArray)
    case AggSignature2(CollectAsSet(), _, Seq(t), _) =>
      new CollectAsSetAggregator(PType.canonical(t))
    case AggSignature2(Collect(), _, Seq(t), _) =>
      new CollectAggregator(t.physicalType)
    case AggSignature2(LinearRegression(), _, _, _) =>
      LinearRegressionAggregator
    case AggSignature2(ApproxCDF(), _, _, _) => new ApproxCDFAggregator
    case AggSignature2(Downsample(), _, Seq(_, _, label), _) => new DownsampleAggregator(label.physicalType.asInstanceOf[PArray])
    case _ => throw new UnsupportedExtraction(aggSig.toString)
  }

  def getPType(aggSig: AggSignature2): PType = getAgg(aggSig).resultType

  def getType(aggSig: AggSignature2): Type = getPType(aggSig).virtualType

  def apply(ir: IR, resultName: String): Aggs = {
    val ab = new ArrayBuilder[InitOp2]()
    val seq = new ArrayBuilder[IR]()
    val let = new ArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, seq, let, ref)
    val initOps = ab.result()
    val aggs = initOps.map(_.aggSig)
    val rt = TTuple(aggs.map(Extract.getType): _*)
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
        val newSig = AggSignature2(
          x.aggSig.op,
          x.aggSig.constructorArgs ++ x.aggSig.initOpArgs.getOrElse(FastSeq.empty),
          x.aggSig.seqOpArgs,
          None)
        ab += InitOp2(i, x.constructorArgs ++ x.initOpArgs.getOrElse[IndexedSeq[IR]](FastIndexedSeq()), newSig)
        seqBuilder += SeqOp2(i, x.seqOpArgs, newSig)
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

        val rt = TDict(key.typ, TTuple(aggs.map(Extract.getType): _*))
        newRef._typ = -rt.elementType

        val aggSig = AggSignature2(Group(), Seq(TVoid), FastSeq(key.typ, TVoid), Some(aggs))
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

        val rt = TArray(TTuple(aggs.map(Extract.getType): _*))
        newRef._typ = -rt.elementType

        val aggSigCheck = AggSignature2(
          AggElementsLengthCheck(),
          knownLength.map(l => FastSeq(l.typ)).getOrElse(FastSeq()) :+ TVoid,
          FastSeq(TInt32()), Some(aggs))
        val aggSig = AggSignature2(AggElements(), FastSeq[Type](), FastSeq(TInt32(), TVoid), Some(aggs))

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
