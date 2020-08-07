package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.{BroadcastRow, Region, UnsafeRow}
import is.hail.asm4s.Code
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.ir.lowering.TableStage
import is.hail.types.TableType
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.virtual.{TArray, TStruct, Type}
import is.hail.rvd.{RVD, RVDCoercer, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.{Extraction, JValue}
import org.json4s.JsonAST.JObject

class PartitionIteratorLongReader(
  val fullRowType: TStruct,
  val contextType: Type,
  bodyPType: Type => PType,
  body: Type => (Region, Any) => Iterator[Long]) extends PartitionReader {

  def rowPType(requestedType: Type): PType = bodyPType(requestedType.asInstanceOf[TStruct])

  def emitStream[C](context: IR,
    requestedType: Type,
    emitter: Emit[C],
    mb: EmitMethodBuilder[C],
    region: StagedRegion,
    env: Emit.E,
    container: Option[AggContainer]): COption[SizedStream] = {

    def emitIR(ir: IR, env: Emit.E = env, region: StagedRegion = region, container: Option[AggContainer] = container): EmitCode =
      emitter.emitWithRegion(ir, mb, region, env, container)

    val eltPType = bodyPType(requestedType)

    COption.fromEmitCode(emitIR(context)).map { contextPC =>
      // FIXME SafeRow.read can only handle address values
      assert(contextPC.pt.isInstanceOf[PStruct])

      val it = mb.newLocal[Iterator[java.lang.Long]]("pilr_it")
      val hasNext = mb.newLocal[Boolean]("pilr_hasNext")
      val next = mb.newLocal[Long]("pilr_next")

      SizedStream.unsized { eltRegion =>
        Stream
          .unfold[Code[Long]](
            (_, k) =>
              Code(
                hasNext := it.get.hasNext,
                hasNext.orEmpty(next := Code.longValue(it.get.next())),
                k(COption(!hasNext, next))),
            setup = Some(
              it := mb.getObject(body(requestedType))
                .invoke[java.lang.Object, java.lang.Object, Iterator[java.lang.Long]]("apply",
                  region.code,
                  Code.invokeScalaObject3[PType, Region, Long, java.lang.Object](UnsafeRow.getClass, "read",
                    mb.getPType(contextPC.pt), region.code, contextPC.tcode[Long]))))
          .map(rv => EmitCode.present(eltPType, Region.loadIRIntermediate(eltPType)(rv)))
      }
    }
  }

  def toJValue: JValue = {
    JObject(
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
  val body: TStruct => (Region, Any) => Iterator[Long]) {

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
      TableStage(globalsIR, p, contextsIR, requestedBody)
    } else {
      getLTVCoercer(ctx).coerce(
        globalsIR,
        contextType, contexts,
        requestedBody)
    }
  }

  def toContextRDD(requestedRowType: TStruct): ContextRDD[Long] =
    ContextRDD(new GenericTableValueRDD(contexts, body(requestedRowType)))

  private[this] var rvdCoercer: RVDCoercer = _

  def getRVDCoercer(ctx: ExecuteContext): RVDCoercer = {
    if (rvdCoercer == null) {
      rvdCoercer = RVD.makeCoercer(
        ctx,
        RVDType(bodyPType(fullTableType.rowType), fullTableType.key),
        1,
        toContextRDD(fullTableType.keyType))
    }
    rvdCoercer
  }

  def toTableValue(ctx: ExecuteContext, requestedType: TableType): TableValue = {
    val requestedRowType = requestedType.rowType
    val requestedRowPType = bodyPType(requestedType.rowType)
    val crdd = toContextRDD(requestedRowType)

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
