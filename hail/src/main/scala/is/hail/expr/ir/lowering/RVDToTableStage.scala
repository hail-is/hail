package is.hail.expr.ir.lowering

import is.hail.annotations.{BroadcastRow, Region, RegionValue}
import is.hail.asm4s._
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.backend.spark.{AnonymousDependency, SparkTaskContext}
import is.hail.expr.ir.{Compile, CompileIterator, GetField, IR, In, Let, MakeStruct, PartitionRVDReader, ReadPartition, StreamRange, ToArray, _}
import is.hail.io.fs.FS
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.rvd.{RVD, RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.types.{RTable, TableType, VirtualTypeWithReq}
import is.hail.types.physical.stypes.PTypeReferenceSingleCodeType
import is.hail.types.physical.{PArray, PStruct}
import is.hail.types.virtual.TStruct
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Dependency, Partition, SparkContext, TaskContext}
import org.json4s.JValue
import org.json4s.JsonAST.JString

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

case class RVDTableReader(rvd: RVD, globals: IR, rt: RTable) extends TableReader {
  lazy val fullType: TableType = TableType(rvd.rowType, rvd.typ.key, globals.typ.asInstanceOf[TStruct])

  override def pathsUsed: Seq[String] = Seq()

  override def partitionCounts: Option[IndexedSeq[Long]] = None

  override def toExecuteIntermediate(ctx: ExecuteContext, requestedType: TableType, dropRows: Boolean): TableExecuteIntermediate = {
    assert(!dropRows)
    val (Some(PTypeReferenceSingleCodeType(globType: PStruct)), f) = Compile[AsmFunction1RegionLong](
      ctx, FastIndexedSeq(), FastIndexedSeq(classInfo[Region]), LongInfo, PruneDeadFields.upcast(ctx, globals, requestedType.globalType))
    val gbAddr = f(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)(ctx.r)

    val globRow = BroadcastRow(ctx, RegionValue(ctx.r, gbAddr), globType)

    val rowEmitType = SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(rvd.rowPType))
    val (Some(PTypeReferenceSingleCodeType(newRowType: PStruct)), rowF) = Compile[AsmFunction2RegionLongLong](
      ctx, FastIndexedSeq(("row", rowEmitType)), FastIndexedSeq(classInfo[Region], LongInfo), LongInfo,
      PruneDeadFields.upcast(ctx, In(0, rowEmitType),
        requestedType.rowType))

    val fsBc = ctx.fsBc
    TableExecuteIntermediate(TableValue(ctx, requestedType, globRow, rvd.mapPartitionsWithIndex(RVDType(newRowType, requestedType.key)) { case (i, ctx, it) =>
      val partF = rowF(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), ctx.partitionRegion)
      it.map { elt => partF(ctx.r, elt) }
    }))
  }

  override def isDistinctlyKeyed: Boolean = false

  def rowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
    VirtualTypeWithReq.subset(requestedType.rowType, rt.rowType)
  }

  def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
    VirtualTypeWithReq.subset(requestedType.globalType, rt.globalType)
  }

  override def toJValue: JValue = JString("RVDTableReader")

  def renderShort(): String = "RVDTableReader"

  override def defaultRender(): String = "RVDTableReader"

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalsType: TStruct): IR =
    PruneDeadFields.upcast(ctx, globals, requestedGlobalsType)

  override def _lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    RVDToTableStage(rvd, globals)
      .upcast(ctx, requestedType)
  }
}

object RVDToTableStage {
  def apply(rvd: RVD, globals: IR): TableStage = {
    TableStage(
      globals = globals,
      partitioner = rvd.partitioner,
      dependency = TableStageDependency.fromRVD(rvd),
      contexts = StreamRange(0, rvd.getNumPartitions, 1),
      body = ReadPartition(_, rvd.rowType, PartitionRVDReader(rvd, "__dummy_uid"))
    )
  }
}

object TableStageToRVD {
  def apply(ctx: ExecuteContext, _ts: TableStage): (BroadcastRow, RVD) = {

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
    val globalsAndBroadcastVals = ts.letBindings.foldRight[IR](baseStruct) { case ((name, value), acc) =>
      Let(name, value, acc)
    }

    val (Some(PTypeReferenceSingleCodeType(gbPType: PStruct)), f) = Compile[AsmFunction1RegionLong](ctx, FastIndexedSeq(), FastIndexedSeq(classInfo[Region]), LongInfo, globalsAndBroadcastVals)
    val gbAddr = f(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)(ctx.r)

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
      val enc = makeContextEnc(baos, ctx.theHailClassLoader)
      enc.writeRegionValue(contextsPType.loadElement(contextsAddr, i))
      enc.flush()
      baos.toByteArray
    }

    val (newRowPType: PStruct, makeIterator) = CompileIterator.forTableStageToRVD(
      ctx,
      decodedContextPType, decodedBcValsPType,
      ts.broadcastVals.map(_._1).foldRight[IR](ts.partition(In(0, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(decodedContextPType))))) { case (bcVal, acc) =>
        Let(bcVal, GetField(In(1, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(decodedBcValsPType))), bcVal), acc)
      })

    val fsBc = ctx.fsBc

    val sparkDeps = ts.dependency
      .deps
      .map(dep => new AnonymousDependency(dep.asInstanceOf[RVDDependency].rvd.crdd.rdd))

    val rdd = new TableStageToRDD(fsBc, sparkContext, encodedContexts, sparkDeps)

    val crdd = ContextRDD.weaken(rdd)
      .cflatMap { case (rvdContext, (encodedContext, idx)) =>
        val decodedContext = makeContextDec(new ByteArrayInputStream(encodedContext), theHailClassLoaderForSparkWorkers)
          .readRegionValue(rvdContext.partitionRegion)
        val decodedBroadcastVals = makeBcDec(new ByteArrayInputStream(encodedBcVals.value), theHailClassLoaderForSparkWorkers)
          .readRegionValue(rvdContext.partitionRegion)
        makeIterator(theHailClassLoaderForSparkWorkers, fsBc.value, SparkTaskContext.get(), rvdContext, decodedContext, decodedBroadcastVals)
          .map(_.longValue())
      }

    (globRow, RVD(RVDType(newRowPType, ts.key), ts.partitioner, crdd))
  }
}

case class TableStageToRDDPartition(data: Array[Byte], index: Int) extends Partition

class TableStageToRDD(
  fsBc: BroadcastValue[FS],
  sc: SparkContext,
  @transient private val collection: Array[Array[Byte]],
  deps: Seq[Dependency[_]])
  extends RDD[(Array[Byte], Int)](sc, deps) {

  override def getPartitions: Array[Partition] = {
    Array.tabulate(collection.length)(i => TableStageToRDDPartition(collection(i), i))
  }

  override def compute(partition: Partition, context: TaskContext): Iterator[(Array[Byte], Int)] = {
    val sp = partition.asInstanceOf[TableStageToRDDPartition]
    Iterator.single((sp.data, sp.index))
  }
}
