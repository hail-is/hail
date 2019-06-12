package is.hail.expr.ir.agg

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.utils._

import scala.language.{existentials, postfixOps}


class UnsupportedExtraction(msg: String) extends Exception(msg)

object TableMapIRNew {
  def apply(tv: TableValue, newRow: IR): TableValue = {
    val typ = tv.typ
    val gType = typ.globalType

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

    val spec = CodecSpec.defaultUncompressed

    // Order of operations:
    // 1. init op on all aggs and write out.
    // 2. load in init op on each partition, seq op over partition, write out.
    // 3. load in partition aggregations, comb op as necessary, write back out.
    // 4. load in partStarts, calculate newRow based on those results.

    val path = Ref(genUID(), TString())

    // 1. init op on all aggs and write out to initPath
    val initPath = HailContext.get.getTemporaryFile(prefix = Some("scan-init"))

    val (_, initF) = ir.CompileWithAggregators2[Long, Long, Unit](
      extracted.aggs,
      "global", gType.physicalType,
      path.name, PString(),
      Begin(FastIndexedSeq(extracted.init, extracted.writeSet(0, path, spec))))

    Region.scoped { region =>
      val globals =
        if (scanInitNeedsGlobals) {
          val rvb = new RegionValueBuilder(region)
          rvb.start(gType.physicalType)
          rvb.addAnnotation(gType, globalsBc.value)
          rvb.end()
        } else
          0
      val pathOff = region.appendString(initPath)
      val f = initF(0)
      Region.scoped { aggRegion =>
        f.newAggState(aggRegion)
        f(region, globals, false, pathOff, false)
      }
    }

    // 2. load in init op on each partition, seq op over partition, write out.
    val _scanPath = HailContext.get.getTemporaryFile(prefix = Some("scan-partition"))

    val (_, readF) = ir.CompileWithAggregators2[Long, Unit](
      extracted.aggs,
      path.name, PString(),
      extracted.readSet(0, path, spec))

    val (_, writeF) = ir.CompileWithAggregators2[Long, Unit](
      extracted.aggs,
      path.name, PString(),
      extracted.writeSet(0, path, spec))

    val (_, eltSeqF) = ir.CompileWithAggregators2[Long, Long, Unit](
      extracted.aggs,
      "global", gType.physicalType,
      "row", typ.rowType.physicalType,
      extracted.seqPerElt)

    val scanPartitionPaths = tv.rvd.collectPerPartition { (i, ctx, it) =>
      val globalRegion = ctx.freshRegion
      val globals =
        if (scanSeqNeedsGlobals) {
          val rvb = new RegionValueBuilder(globalRegion)
          rvb.start(gType.physicalType)
          rvb.addAnnotation(gType, globalsBc.value)
          rvb.end()
        } else
          0

      val aggRegion = ctx.freshRegion

      val resultFile = _scanPath + "-%04d".format(i)
      val initFileOff = globalRegion.appendString(initPath)
      val resultFileOff = globalRegion.appendString(resultFile)

      val init = readF(i)
      val write = writeF(i)
      val seq = eltSeqF(i)
      init.newAggState(aggRegion)

      init(globalRegion, initFileOff, false)
      seq.setAggState(aggRegion, init.getAggOffset())

      it.foreach { rv =>
        seq(rv.region, globals, false, rv.offset, false)
        ctx.region.clear()
      }

      write.setAggState(aggRegion, seq.getAggOffset())
      write(globalRegion, resultFileOff, false)
      resultFile
    }

    // 3. load in partition aggregations, comb op as necessary, write back out.
    val (_, readF2) = ir.CompileWithAggregators2[Long, Unit](
      extracted.aggs ++ extracted.aggs,
      path.name, PString(),
      extracted.readSet(0, path, spec))

    val path2 = Ref(genUID(), TString())
    val (_, combOpF) = ir.CompileWithAggregators2[Long, Long, Unit](
      extracted.aggs ++ extracted.aggs,
      path.name, PString(), // next partition
      path2.name, PString(), // result file
      Begin(
        extracted.readSet(1, path, spec) +:
        Array.tabulate(nAggs)(i => CombOp2(i, nAggs + i, extracted.aggs(i))) :+
        extracted.writeSet(0, path2, spec)))

    val resultPaths = Region.scoped { region =>
      Region.scoped { aggRegion =>
        val read = readF2(0)
        val combOp = combOpF(0)
        read.newAggState(aggRegion)

        var partPath = region.appendString(initPath)

        read(region, partPath, false)
        combOp.setAggState(aggRegion, read.getAggOffset())

        val _resultPath = HailContext.get.getTemporaryFile(prefix = Some("scan-result"))
        initPath +: Array.tabulate(scanPartitionPaths.length - 1) { i =>
          val destPath = _resultPath + "-%04d".format(i)
          val pathOff = region.appendString(destPath)
          combOp(region, partPath, false, pathOff, false)
          partPath = pathOff
          destPath
        }
      }
    }

    // 4. load in partStarts, calculate newRow based on those results.

    val (rTyp, f) = ir.CompileWithAggregators2[Long, Long, Long](
      extracted.aggs,
      "global", gType.physicalType,
      "row", typ.rowType.physicalType,
      Let(scanRef, extracted.results, extracted.postAggIR))
    assert(rTyp.virtualType == typ.rowType)

    val itF = { (i: Int, ctx: RVDContext, partitionAggs: String, it: Iterator[RegionValue]) =>
      val globalRegion = ctx.freshRegion
      val rvb = new RegionValueBuilder(globalRegion)
      val globals =
        if (rowIterationNeedsGlobals || scanSeqNeedsGlobals) {
          rvb.set(globalRegion)
          rvb.start(gType.physicalType)
          rvb.addAnnotation(gType, globalsBc.value)
          rvb.end()
        } else
          0

      val path = globalRegion.appendString(partitionAggs)
      val aggRegion = ctx.freshRegion

      val read = readF(i)
      val newRow = f(i)
      val seq = eltSeqF(i)
      read.newAggState(aggRegion)

      read(globalRegion, path, false)
      newRow.setAggState(aggRegion, read.getAggOffset())
      seq.setAggState(aggRegion, read.getAggOffset())

      it.map { rv =>
        val off = newRow(rv.region, globals, false, rv.offset, false)
        seq(rv.region, globals, false, rv.offset, false)
        rv.setOffset(off)
        rv
      }
    }

    tv.copy(
      typ = typ,
      rvd = tv.rvd.mapPartitionsWithIndexAndValue(RVDType(rTyp.asInstanceOf[PStruct], typ.key), resultPaths, itF))
  }
}

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[AggSignature]) {
  val typ: PTuple = PTuple(aggs.map(Extract.getPType))
  val nAggs: Int = aggs.length

  def readSet(i: Int, path: IR, spec: CodecSpec): IR =
    ReadAggs(i * nAggs, path, spec, aggs)

  def writeSet(i: Int, path: IR, spec: CodecSpec): IR =
    WriteAggs(i * nAggs, path, spec, aggs)

  def eltOp: IR = seqPerElt

  def results: IR = ResultOp2(0, aggs)
}

object Extract {
  def addLets(ir: IR, lets: Array[AggLet]): IR = {
    assert(lets.areDistinct())
    lets.foldRight[IR](ir) { case (al, comb) => Let(al.name, al.value, comb)}
  }

  def compatible(sig1: AggSignature, sig2: AggSignature): Boolean = (sig1.op, sig2.op) match {
    case (AggElements2(nestedAggs1), AggElements2(nestedAggs2)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case (AggElementsLengthCheck2(nestedAggs1, _), AggElements2(nestedAggs2)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case (AggElements2(nestedAggs1), AggElementsLengthCheck2(nestedAggs2, _)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case (AggElementsLengthCheck2(nestedAggs1, _), AggElementsLengthCheck2(nestedAggs2, _)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case _ => sig1 == sig2
  }

  def getAgg(aggSig: AggSignature): StagedRegionValueAggregator = aggSig match {
    case AggSignature(Sum(), _, _, Seq(t)) =>
      new SumAggregator(t.physicalType)
    case AggSignature(Count(), _, _, Seq(t)) =>
      new SumAggregator(t.physicalType)
    case AggSignature(AggElementsLengthCheck2(nestedAggs, knownLength), _, _, _) =>
      new ArrayElementLengthCheckAggregator(nestedAggs.map(getAgg).toArray, knownLength)
    case AggSignature(AggElements2(nestedAggs), _, _, _) =>
      new ArrayElementwiseOpAggregator(nestedAggs.map(getAgg).toArray)
    case AggSignature(PrevNonnull(), _, _, Seq(t)) =>
      new PrevNonNullAggregator(t.physicalType)
    case _ => throw new UnsupportedExtraction("foo")
  }

  def getPType(aggSig: AggSignature): PType = getAgg(aggSig).resultType

  def getType(aggSig: AggSignature): Type = getPType(aggSig).virtualType

  def apply(ir: IR, resultName: String): Aggs = {
    val ab = new ArrayBuilder[(AggSignature, IndexedSeq[IR])]()
    val seq = new ArrayBuilder[IR]()
    val let = new ArrayBuilder[AggLet]()
    val ref = Ref(resultName, null)
    val postAgg = extract(ir, ab, seq, let, ref)
    val (aggs, initArgs) = ab.result().unzip
    val rt = TTuple(aggs.map(Extract.getType): _*)
    ref._typ = rt

    val initOps = Array.tabulate(initArgs.length)(i => InitOp2(i, initArgs(i), aggs(i)))
    Aggs(postAgg, Begin(initOps), Begin(seq.result()), aggs)
  }

  private def extract(ir: IR, ab: ArrayBuilder[(AggSignature, IndexedSeq[IR])], seqBuilder: ArrayBuilder[IR], letBuilder: ArrayBuilder[AggLet], result: IR): IR = {
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
        ab += x.aggSig -> (x.constructorArgs ++ x.initOpArgs.getOrElse[IndexedSeq[IR]](FastIndexedSeq()))
        seqBuilder += SeqOp2(i, x.seqOpArgs, x.aggSig)
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

        val (dependent, independent) = newLet.result().partition(l => Mentions(l.value, name))
        letBuilder ++= independent
        seqBuilder += ArrayFor(array, name, addLets(Begin(newSeq.result()), dependent))
        transformed

      case AggGroupBy(key, aggIR, _) =>
        throw new UnsupportedExtraction("group by")

      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, _) =>
        val newAggs = new ArrayBuilder[(AggSignature, IndexedSeq[IR])]()
        val newSeq = new ArrayBuilder[IR]()
        val newLet = new ArrayBuilder[AggLet]()
        val newRef = Ref(genUID(), null)
        val transformed = this.extract(aggBody, newAggs, newSeq, newLet, newRef)

        val (dependent, independent) = newLet.result().partition(l => Mentions(l.value, elementName))
        letBuilder ++= independent

        val i = ab.length
        val (aggs, inits) = newAggs.result().unzip
        val rt = TArray(TTuple(aggs.map(Extract.getType): _*))
        newRef._typ = -rt.elementType
        val initArgs = knownLength.map(FastIndexedSeq(_)).getOrElse[IndexedSeq[IR]](FastIndexedSeq()) ++ inits.flatten.toFastIndexedSeq

        val aggSigCheck = AggSignature(AggElementsLengthCheck2(aggs, knownLength.isDefined), FastSeq[Type](),
          Some(initArgs.map(_.typ)), FastSeq(TInt32()))
        val aggSig = AggSignature(AggElements2(aggs), FastSeq[Type](), None, FastSeq(TInt32(), TVoid))

        val aRef = Ref(genUID(), a.typ)
        val iRef = Ref(genUID(), TInt32())

        ab += aggSigCheck -> initArgs
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