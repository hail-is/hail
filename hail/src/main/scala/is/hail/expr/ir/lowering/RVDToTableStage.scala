package is.hail.expr.ir.lowering

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.annotations.{BroadcastRow, Region, RegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.{Compile, CompileIterator, ExecuteContext, GetField, IR, In, Let, MakeStruct, PartitionRVDReader, ReadPartition, StreamRange, ToArray, _}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.rvd.{RVD, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.{PArray, PStruct, PTypeReferenceSingleCodeType}
import is.hail.utils.{FastIndexedSeq, FastSeq}

object RVDToTableStage {
  def apply(rvd: RVD, globals: IR): TableStage = {
    TableStage(
      globals = globals,
      partitioner = rvd.partitioner,
      dependency = TableStageDependency.fromRVD(rvd),
      contexts = StreamRange(0, rvd.getNumPartitions, 1),
      body = ReadPartition(_, rvd.rowType, PartitionRVDReader(rvd))
    )
  }
}

object TableStageToRVD {
  def apply(ctx: ExecuteContext, _ts: TableStage, relationalBindings: Map[String, IR]): (BroadcastRow, RVD) = {

    val ts = TableStage(letBindings = _ts.letBindings,
      broadcastVals = _ts.broadcastVals,
      globals = _ts.globals,
      partitioner = _ts.partitioner,
      dependency = _ts.dependency,
      contexts = mapIR(_ts.contexts) { c => MakeStruct(FastSeq("context" -> c)) },
      partition = { ctx: Ref => _ts.partition(GetField(ctx, "context")) })

    val sparkContext = ctx.backend
      .asSpark("TableStageToRVD")
      .sc

    val baseStruct = MakeStruct(FastSeq(
      ("globals", ts.globals),
      ("broadcastVals", MakeStruct(ts.broadcastVals)),
      ("contexts", ToArray(ts.contexts))))
    val globalsAndBroadcastVals = LowerToCDA.substLets(ts.letBindings.foldRight[IR](baseStruct) { case ((name, value), acc) =>
      Let(name, value, acc)
    }, relationalBindings)

    val (Some(PTypeReferenceSingleCodeType(gbPType: PStruct)), f) = Compile[AsmFunction1RegionLong](ctx, FastIndexedSeq(), FastIndexedSeq(classInfo[Region]), LongInfo, globalsAndBroadcastVals)
    val gbAddr = f(ctx.fs, 0, ctx.r)(ctx.r)

    val globPType = gbPType.fieldType("globals").asInstanceOf[PStruct]
    val globRow = BroadcastRow(ctx, RegionValue(ctx.r, gbPType.loadField(gbAddr, 0)), globPType)

    val bcValsPType = gbPType.fieldType("broadcastVals")

    val bcValsSpec = TypedCodecSpec(bcValsPType, BufferSpec.wireSpec)
    val encodedBcVals = sparkContext.broadcast(bcValsSpec.encodeValue(ctx, bcValsPType, gbPType.loadField(gbAddr, 1)))
    val (decodedBcValsPType: PStruct, makeBcDec) = bcValsSpec.buildDecoder(ctx, bcValsPType.virtualType)

    val contextsPType = gbPType.fieldType("contexts").asInstanceOf[PArray]
    val contextPType = contextsPType.elementType
    val contextSpec = TypedCodecSpec(contextPType, BufferSpec.wireSpec)
    val contextsAddr = gbPType.loadField(gbAddr, 2)
    val nContexts = contextsPType.loadLength(contextsAddr)

    val (decodedContextPType: PStruct, makeContextDec) = contextSpec.buildDecoder(ctx, contextPType.virtualType)

    val makeContextEnc = contextSpec.buildEncoder(ctx, contextPType)
    val encodedContexts = Array.tabulate(nContexts) { i =>
      assert(contextsPType.isElementDefined(contextsAddr, i))
      val baos = new ByteArrayOutputStream()
      val enc = makeContextEnc(baos)
      enc.writeRegionValue(contextsPType.loadElement(contextsAddr, i))
      enc.flush()
      baos.toByteArray
    }

    val (newRowPType: PStruct, makeIterator) = CompileIterator.forTableStageToRVD(
      ctx,
      decodedContextPType, decodedBcValsPType,
      LowerToCDA.substLets(ts.broadcastVals.map(_._1).foldRight[IR](ts.partition(In(0, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(decodedContextPType))))) { case (bcVal, acc) =>
        Let(bcVal, GetField(In(1, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(decodedBcValsPType))), bcVal), acc)
      }, relationalBindings))


    val fsBc = ctx.fsBc
    val crdd = ContextRDD.weaken(sparkContext
      .parallelize(encodedContexts.zipWithIndex, numSlices = nContexts))
      .cflatMap { case (rvdContext, (encodedContext, idx)) =>
        val decodedContext = makeContextDec(new ByteArrayInputStream(encodedContext))
          .readRegionValue(rvdContext.partitionRegion)
        val decodedBroadcastVals = makeBcDec(new ByteArrayInputStream(encodedBcVals.value))
          .readRegionValue(rvdContext.partitionRegion)
        makeIterator(fsBc.value, idx, rvdContext, decodedContext, decodedBroadcastVals)
          .map(_.longValue())
      }

    (globRow, RVD(RVDType(newRowPType, ts.key), ts.partitioner, crdd))
  }
}