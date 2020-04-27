package is.hail.expr.ir

import is.hail.annotations.{BroadcastRow, Region}
import is.hail.asm4s.Value
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.ir.lowering.TableStage
import is.hail.expr.types.TableType
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual.{TStruct, Type}
import is.hail.rvd.{RVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

class PartitionIteratorLongReader(
  val fullRowType: TStruct,
  val contextType: Type,
  bodyPType: TStruct => PType,
  body: TStruct => (Region, Any) => Iterator[Long]) extends PartitionReader {

  def rowPType(requestedType: Type): PType = bodyPType(requestedType.asInstanceOf[TStruct])

  def emitStream[C](context: IR,
    requestedType: Type,
    emitter: Emit[C],
    mb: EmitMethodBuilder[C],
    region: Value[Region],
    env0: Emit.E,
    container: Option[AggContainer]): COption[SizedStream] = {
    ???
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
  val globals: TStruct => Row,
  val contextType: Type,
  val contexts: IndexedSeq[Any],
  val bodyPType: TStruct => PStruct,
  val body: TStruct => (Region, Any) => Iterator[Long]) {

  def toTableStage(requestedType: TableType): TableStage = ???

  def toContextRDD(requestedRowType: TStruct): ContextRDD[Long] =
    ContextRDD(new GenericTableValueRDD(contexts, body(requestedRowType)))

  def toTableValue(ctx: ExecuteContext, requestedType: TableType): TableValue = {
    assert(bodyPType(requestedType.rowType) == requestedType.canonicalRowPType)
    assert(bodyPType(requestedType.keyType) == requestedType.canonicalRVDType.kType)

    TableValue(ctx,
      requestedType,
      BroadcastRow(ctx, globals(requestedType.globalType), requestedType.globalType),
      RVD.coerce(ctx, requestedType.canonicalRVDType,
        toContextRDD(requestedType.rowType),
        toContextRDD(requestedType.keyType)))
  }
}
