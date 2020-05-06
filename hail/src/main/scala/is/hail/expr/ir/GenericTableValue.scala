package is.hail.expr.ir

import is.hail.utils._
import is.hail.annotations.{BroadcastRow, Region, UnsafeRow}
import is.hail.asm4s.{Code, Value}
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.types.TableType
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual.{TArray, TStruct, Type}
import is.hail.rvd.{RVD, RVDContext, RVDPartitioner, RVDType}
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
    region: Value[Region],
    env: Emit.E,
    container: Option[AggContainer]): COption[SizedStream] = {

    def emitIR(ir: IR, env: Emit.E = env, region: Value[Region] = region, container: Option[AggContainer] = container): EmitCode =
      emitter.emitWithRegion(ir, mb, region, env, container)

    val eltPType = bodyPType(requestedType)

    COption.fromEmitCode(emitIR(context)).map { contextPC =>
      // FIXME SafeRow.read can only handle address values
      assert(contextPC.pt.isInstanceOf[PStruct])

      val it = mb.newLocal[Iterator[java.lang.Long]]("pilr_it")

      SizedStream.unsized(Stream.unfold[Code[Long]](
        (_, k) => k(COption(
          !it.get.hasNext,
          Code.longValue(it.get.next()))),
        setup = Some(
          it := mb.getObject(body(requestedType))
            .invoke[java.lang.Object, java.lang.Object, Iterator[java.lang.Long]]("apply",
              region,
              Code.invokeScalaObject3[PType, Region, Long, java.lang.Object](UnsafeRow.getClass, "read",
                mb.getPType(contextPC.pt), region, contextPC.tcode[Long]))))
      .map(rv => EmitCode.present(eltPType, Region.loadIRIntermediate(eltPType)(rv))))
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

class GenericTableValue(
  val t: TableType,
  val partitioner: RVDPartitioner,
  val globals: TStruct => Row,
  val contextType: Type,
  val contexts: IndexedSeq[Any],
  val bodyPType: TStruct => PStruct,
  val body: TStruct => (Region, Any) => Iterator[Long]) {

  def toLoweredTableValue(requestedType: TableType): LoweredTableReader = {
    LoweredTableReader(
      Literal(requestedType.globalType, globals(requestedType.globalType)),
      partitioner,
      ToStream(Literal(TArray(contextType), contexts)),
      (ctx: IR) => ReadPartition(ctx,
        requestedType.rowType,
        new PartitionIteratorLongReader(
          t.rowType, contextType,
          (requestedType: Type) => bodyPType(requestedType.asInstanceOf[TStruct]),
          (requestedType: Type) => body(requestedType.asInstanceOf[TStruct]))))
  }

  def toContextRDD(requestedRowType: TStruct): ContextRDD[Long] =
    ContextRDD(new GenericTableValueRDD(contexts, body(requestedRowType)))

  def toTableValue(ctx: ExecuteContext, requestedType: TableType): TableValue = {
    TableValue(ctx,
      requestedType,
      BroadcastRow(ctx, globals(requestedType.globalType), requestedType.globalType),
      RVD(
        RVDType(bodyPType(requestedType.rowType), t.key),
        partitioner,
        toContextRDD(requestedType.rowType)))
  }
}
