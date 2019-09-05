package is.hail.expr.ir.agg

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue}
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{BufferSpec, CodecSpec, CodecSpec2}
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.utils._

import scala.collection.mutable
import scala.language.{existentials, postfixOps}

object TableMapIRNew {

  def apply(tv: TableValue, newRow: IR): TableValue = {
    val typ = tv.typ

    val scanRef = genUID()
    val extracted = Extract.apply(CompileWithAggregators.liftScan(newRow), scanRef)
    val nAggs = extracted.nAggs

    if (extracted.aggs.isEmpty)
      throw new UnsupportedExtraction("no scans to extract in TableMapRows")

    val scanInitNeedsGlobals = Mentions(extracted.init, "global")
    val scanSeqNeedsGlobals = Mentions(extracted.seqPerElt, "global")
    val rowIterationNeedsGlobals = Mentions(extracted.postAggIR, "global")

    val globalsBc =
      if (rowIterationNeedsGlobals || scanInitNeedsGlobals || scanSeqNeedsGlobals)
        tv.globals.broadcast
      else
        null

    val spec = CodecSpec.defaultUncompressedBuffer

    // Order of operations:
    // 1. init op on all aggs and serialize to byte array.
    // 2. load in init op on each partition, seq op over partition, serialize.
    // 3. load in partition aggregations, comb op as necessary, serialize.
    // 4. load in partStarts, calculate newRow based on those results.

    val (_, initF) = ir.CompileWithAggregators2[Long, Unit](
      extracted.aggs,
      "global", tv.globals.t,
      Begin(FastIndexedSeq(extracted.init, extracted.serializeSet(0, 0, spec))))

    val (_, eltSeqF) = ir.CompileWithAggregators2[Long, Long, Unit](
      extracted.aggs,
      "global", Option(globalsBc).map(_.value.t).getOrElse(PStruct()),
      "row", typ.rowType.physicalType,
      extracted.eltOp())

    val read = extracted.deserialize(spec)
    val write = extracted.serialize(spec)
    val combOpF = extracted.combOpF(spec)

    val (rTyp, f) = ir.CompileWithAggregators2[Long, Long, Long](
      extracted.aggs,
      "global", Option(globalsBc).map(_.value.t).getOrElse(PStruct()),
      "row", typ.rowType.physicalType,
      Let(scanRef, extracted.results, extracted.postAggIR))
    assert(rTyp.virtualType == newRow.typ)

    // 1. init op on all aggs and write out to initPath
    val initAgg = Region.scoped { aggRegion =>
      Region.scoped { fRegion =>
        val init = initF(0, fRegion)
        init.newAggState(aggRegion)
        init(fRegion, tv.globals.value.offset, false)
        init.getSerializedAgg(0)
      }
    }

    // 2. load in init op on each partition, seq op over partition, write out.
    val scanPartitionAggs = SpillingCollectIterator(tv.rvd.mapPartitionsWithIndex { (i, ctx, it) =>
      val globalRegion = ctx.freshRegion
      val globals = if (scanSeqNeedsGlobals) globalsBc.value.readRegionValue(globalRegion) else 0

      Region.smallScoped { aggRegion =>
        val seq = eltSeqF(i, globalRegion)

        seq.setAggState(aggRegion, read(aggRegion, initAgg))
        it.foreach { rv =>
          seq(rv.region, globals, false, rv.offset, false)
          ctx.region.clear()
        }
        Iterator.single(write(aggRegion, seq.getAggOffset()))
      }
    }, HailContext.get.flags.get("max_leader_scans").toInt)


    // 3. load in partition aggregations, comb op as necessary, write back out.
    val partAggs = scanPartitionAggs.scanLeft(initAgg)(combOpF)
    val scanAggCount = tv.rvd.getNumPartitions
    val partitionIndices = new Array[Long](scanAggCount)
    val scanAggsPerPartitionFile = HailContext.get.getTemporaryFile()
    HailContext.get.sFS.writeFileNoCompression(scanAggsPerPartitionFile) { os =>
      partAggs.zipWithIndex.foreach { case (x, i) =>
        if (i < scanAggCount) {
          partitionIndices(i) = os.getPos
          os.writeInt(x.length)
          os.write(x, 0, x.length)
          os.hflush()
        }
      }
    }

    val bcFS = HailContext.get.bcFS

    // 4. load in partStarts, calculate newRow based on those results.
    val itF = { (i: Int, ctx: RVDContext, filePosition: Long, it: Iterator[RegionValue]) =>
      val globalRegion = ctx.freshRegion
      val globals = if (rowIterationNeedsGlobals || scanSeqNeedsGlobals)
        globalsBc.value.readRegionValue(globalRegion)
      else
        0
      val partitionAggs = bcFS.value.readFileNoCompression(scanAggsPerPartitionFile) { is =>
        is.seek(filePosition)
        val aggSize = is.readInt()
        val partAggs = new Array[Byte](aggSize)
        var nread = is.read(partAggs, 0, aggSize)
        var r = nread
        while (r > 0 && nread < aggSize) {
          r = is.read(partAggs, nread, aggSize - nread)
          if (r > 0) nread += r
        }
        if (nread != aggSize) {
          fatal(s"aggs read wrong number of bytes: $nread vs $aggSize")
        }
        partAggs
      }

      val aggRegion = ctx.freshRegion
      val newRow = f(i, globalRegion)
      val seq = eltSeqF(i, globalRegion)
      var aggOff = read(aggRegion, partitionAggs)

      it.map { rv =>
        newRow.setAggState(aggRegion, aggOff)
        val off = newRow(rv.region, globals, false, rv.offset, false)
        seq.setAggState(aggRegion, newRow.getAggOffset())
        seq(rv.region, globals, false, rv.offset, false)
        aggOff = seq.getAggOffset()
        rv.setOffset(off)
        rv
      }
    }
    tv.copy(
      typ = typ.copy(rowType = rTyp.virtualType.asInstanceOf[TStruct]),
      rvd = tv.rvd.mapPartitionsWithIndexAndValue(RVDType(rTyp.asInstanceOf[PStruct], typ.key), partitionIndices, itF))
  }
}

class UnsupportedExtraction(msg: String) extends Exception(msg)

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[AggSignature2]) {
  val typ: PTuple = PTuple(aggs.map(Extract.getPType): _*)
  val nAggs: Int = aggs.length

  def isCommutative: Boolean = {
    def aggCommutes(agg: AggSignature2): Boolean = agg.nested.forall(_.forall(aggCommutes)) && (agg.op match {
      case Take() | Collect() | PrevNonnull() | TakeBy() => false
      case _ => true
    })
    aggs.forall(aggCommutes)
  }

  def deserializeSet(i: Int, i2: Int, spec: BufferSpec): IR =
    DeserializeAggs(i * nAggs, i2, spec, aggs)

  def serializeSet(i: Int, i2: Int, spec: BufferSpec): IR =
    SerializeAggs(i * nAggs, i2, spec, aggs)

  def eltOp(optimize: Boolean = true): IR = if (optimize) Optimize(seqPerElt) else seqPerElt

  def deserialize(spec: BufferSpec): ((Region, Array[Byte]) => Long) = {
    val (_, f) = ir.CompileWithAggregators2[Unit](
      aggs, ir.DeserializeAggs(0, 0, spec, aggs))

    { (aggRegion: Region, bytes: Array[Byte]) =>
      val f2 = f(0, aggRegion);
      f2.newAggState(aggRegion)
      f2.setSerializedAgg(0, bytes)
      f2(aggRegion)
      f2.getAggOffset()
    }
  }

  def serialize(spec: BufferSpec): (Region, Long) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators2[Unit](
      aggs, ir.SerializeAggs(0, 0, spec, aggs))

    { (aggRegion: Region, off: Long) =>
      val f2 = f(0, aggRegion);
      f2.setAggState(aggRegion, off)
      f2(aggRegion)
      f2.getSerializedAgg(0)
    }
  }

  def combOpF(spec: BufferSpec): (Array[Byte], Array[Byte]) => Array[Byte] = {
    val (_, f) = ir.CompileWithAggregators2[Unit](
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
    case AggSignature2(AggElementsLengthCheck(), initOpArgs, _, Some(nestedAggs)) =>
      val knownLength = initOpArgs.length == 2
      new ArrayElementLengthCheckAggregator(nestedAggs.map(getAgg).toArray, knownLength)
    case AggSignature2(AggElements(), _, _, Some(nestedAggs)) =>
      new ArrayElementwiseOpAggregator(nestedAggs.map(getAgg).toArray)
    case AggSignature2(PrevNonnull(), _, Seq(t), _) =>
      new PrevNonNullAggregator(t.physicalType)
    case AggSignature2(Group(), _, Seq(kt, TVoid), Some(nestedAggs)) =>
      new GroupedAggregator(PType.canonical(kt), nestedAggs.map(getAgg).toArray)
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
        throw new UnsupportedExtraction("array agg")
      case x: ArrayAggScan =>
        throw new UnsupportedExtraction("array scan")
      case _ => MapIR(extract)(ir)
    }
  }
}
