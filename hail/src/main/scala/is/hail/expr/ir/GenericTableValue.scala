package is.hail.expr.ir

import is.hail.annotations.{BroadcastRow, Region}
import is.hail.asm4s.{Code, CodeLabel, Settable, Value}
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.stypes.interfaces.{SStream, SStreamValue}
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.virtual.{TArray, TStruct, Type}
import is.hail.types.{TableType, TypeWithRequiredness}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, TaskContext}
import org.json4s.JsonAST.{JObject, JString}
import org.json4s.{Extraction, JValue}

class PartitionIteratorLongReader(
  val fullRowType: TStruct,
  val contextType: Type,
  bodyPType: Type => PType,
  body: Type => (Region, FS, Any) => Iterator[Long]) extends PartitionReader {

  def rowRequiredness(requestedType: Type): TypeWithRequiredness = {
    val tr = TypeWithRequiredness.apply(requestedType)
    tr.fromPType(bodyPType(requestedType.asInstanceOf[TStruct]))
    tr
  }

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    context: EmitCode,
    partitionRegion: Value[Region],
    requestedType: Type): IEmitCode = {

    val eltPType = bodyPType(requestedType)
    val mb = cb.emb

    context.toI(cb).map(cb) { contextPC =>
      val ctxJavaValue = UtilFunctions.svalueToJavaValue(cb, partitionRegion, contextPC)
      val region = mb.genFieldThisRef[Region]("pilr_region")
      val it = mb.genFieldThisRef[Iterator[java.lang.Long]]("pilr_it")
      val rv = mb.genFieldThisRef[Long]("pilr_rv")

      val producer = new StreamProducer {
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder): Unit = {
          cb.assign(it, cb.emb.getObject(body(requestedType))
            .invoke[java.lang.Object, java.lang.Object, java.lang.Object, Iterator[java.lang.Long]](
              "apply", region, cb.emb.getFS, ctxJavaValue))
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!it.get.hasNext,
            cb.goto(LendOfStream))
          cb.assign(rv, Code.longValue(it.get.next()))

          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb, eltPType.loadCheapSCode(cb, rv)))

        override def close(cb: EmitCodeBuilder): Unit = {}
      }

      SStreamValue(SStream(producer.element.emitType), producer)
    }
  }

  def toJValue: JValue = {
    JObject(
      "category" -> JString("PartitionIteratorLongReader"),
      "fullRowType" -> Extraction.decompose(fullRowType)(PartitionReader.formats),
      "contextType" -> Extraction.decompose(contextType)(PartitionReader.formats))
  }
}

class GenericTableValueRDDPartition(
  val index: Int,
  val context: Any
) extends Partition

class GenericTableValueRDD(
  @transient val contexts: IndexedSeq[Any],
  body: (Region, Any) => Iterator[Long]
) extends RDD[RVDContext => Iterator[Long]](SparkBackend.sparkContext("GenericTableValueRDD"), Nil) {
  def getPartitions: Array[Partition] = contexts.zipWithIndex.map { case (c, i) =>
    new GenericTableValueRDDPartition(i, c)
  }.toArray

  def compute(split: Partition, context: TaskContext): Iterator[RVDContext => Iterator[Long]] = {
    Iterator.single { (rvdCtx: RVDContext) =>
      body(rvdCtx.region, split.asInstanceOf[GenericTableValueRDDPartition].context)
    }
  }
}

abstract class LoweredTableReaderCoercer {
  def coerce(globals: IR,
    contextType: Type,
    contexts: IndexedSeq[Any],
    body: IR => IR): TableStage
}

class GenericTableValue(
  val fullTableType: TableType,
  val partitioner: Option[RVDPartitioner],
  val globals: TStruct => Row,
  val contextType: Type,
  var contexts: IndexedSeq[Any],
  val bodyPType: TStruct => PStruct,
  val body: TStruct => (Region, FS, Any) => Iterator[Long]) {

  var ltrCoercer: LoweredTableReaderCoercer = _
  def getLTVCoercer(ctx: ExecuteContext): LoweredTableReaderCoercer = {
    if (ltrCoercer == null) {
      ltrCoercer = LoweredTableReader.makeCoercer(
        ctx,
        fullTableType.key,
        1,
        contextType,
        contexts,
        fullTableType.keyType,
        bodyPType,
        body)
    }
    ltrCoercer
  }

  def toTableStage(ctx: ExecuteContext, requestedType: TableType): TableStage = {
    val globalsIR = Literal(requestedType.globalType, globals(requestedType.globalType))
    val requestedBody: (IR) => (IR) = (ctx: IR) => ReadPartition(ctx,
      requestedType.rowType,
      new PartitionIteratorLongReader(
        fullTableType.rowType, contextType,
        (requestedType: Type) => bodyPType(requestedType.asInstanceOf[TStruct]),
        (requestedType: Type) => body(requestedType.asInstanceOf[TStruct])))
    var p: RVDPartitioner = null
    partitioner match {
      case Some(partitioner) =>
        p = partitioner
      case None if requestedType.key.isEmpty =>
        p = RVDPartitioner.unkeyed(contexts.length)
      case None =>
    }
    if (p != null) {
      val contextsIR = ToStream(Literal(TArray(contextType), contexts))
      TableStage(globalsIR, p, TableStageDependency.none, contextsIR, requestedBody)
    } else {
      getLTVCoercer(ctx).coerce(
        globalsIR,
        contextType, contexts,
        requestedBody)
    }
  }

  def toContextRDD(fs: FS, requestedRowType: TStruct): ContextRDD[Long] = {
    val localBody = body(requestedRowType)
    ContextRDD(new GenericTableValueRDD(contexts, localBody(_, fs, _)))
  }

  private[this] var rvdCoercer: RVDCoercer = _

  def getRVDCoercer(ctx: ExecuteContext): RVDCoercer = {
    if (rvdCoercer == null) {
      rvdCoercer = RVD.makeCoercer(
        ctx,
        RVDType(bodyPType(fullTableType.rowType), fullTableType.key),
        1,
        toContextRDD(ctx.fs, fullTableType.keyType))
    }
    rvdCoercer
  }

  def toTableValue(ctx: ExecuteContext, requestedType: TableType): TableValue = {
    val requestedRowType = requestedType.rowType
    val requestedRowPType = bodyPType(requestedType.rowType)
    val crdd = toContextRDD(ctx.fs, requestedRowType)

    val rvd = partitioner match {
      case Some(partitioner) =>
        RVD(
          RVDType(requestedRowPType, fullTableType.key),
          partitioner,
          crdd)
      case None if requestedType.key.isEmpty =>
        RVD(
          RVDType(requestedRowPType, fullTableType.key),
          RVDPartitioner.unkeyed(contexts.length),
          crdd)
      case None =>
        getRVDCoercer(ctx).coerce(RVDType(requestedRowPType, fullTableType.key), crdd)
    }

    TableValue(ctx,
      requestedType,
      BroadcastRow(ctx, globals(requestedType.globalType), requestedType.globalType),
      rvd)
  }
}
