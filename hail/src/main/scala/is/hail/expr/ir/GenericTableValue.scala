package is.hail.expr.ir

import cats.syntax.all._
import is.hail.annotations.{BroadcastRow, Region}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.expr.ir.lowering.{MonadLower, TableStage, TableStageDependency}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.PStruct
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SStackStruct, SStackStructValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SStreamValue, primitive}
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.types.virtual._
import is.hail.types.{RStruct, TableType, TypeWithRequiredness}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, TaskContext}
import org.json4s.JsonAST.{JObject, JString}
import org.json4s.{Extraction, JValue}

import scala.language.higherKinds

class PartitionIteratorLongReader(
  rowType: TStruct,
  override val uidFieldName: String,
  override val contextType: TStruct,
  bodyPType: TStruct => PStruct,
  body: TStruct => (Region, HailClassLoader, FS, Any) => Iterator[Long]
) extends PartitionReader {
  assert(contextType.hasField("partitionIndex"))
  assert(contextType.fieldType("partitionIndex") == TInt32)

  override lazy val fullRowType: TStruct =
    rowType.insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64)))

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val tr = TypeWithRequiredness(requestedType).asInstanceOf[RStruct]
    tr.fromPType(bodyPType(requestedType))
    tr
  }

  override def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName)

    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType

    val eltPType = bodyPType(concreteType)
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)))

    context.toI(cb).map(cb) { case _ctxStruct: SBaseStructValue =>
      val ctxStruct = cb.memoizeField(_ctxStruct, "ctxStruct")
      val partIdx = cb.memoizeField(_ctxStruct.loadField(cb, "partitionIndex").get(cb).asInt.value.toL)
      val rowIdx = mb.genFieldThisRef[Long]("pnr_rowidx")
      val region = mb.genFieldThisRef[Region]("pilr_region")
      val it = mb.genFieldThisRef[Iterator[java.lang.Long]]("pilr_it")
      val rv = mb.genFieldThisRef[Long]("pilr_rv")

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb

        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          val ctxJavaValue = UtilFunctions.svalueToJavaValue(cb, partitionRegion, ctxStruct)
          cb.assign(it, cb.emb.getObject(body(requestedType))
            .invoke[java.lang.Object, java.lang.Object, java.lang.Object, java.lang.Object, Iterator[java.lang.Long]](
              "apply", region, cb.emb.getHailClassLoader, cb.emb.getFS, ctxJavaValue))
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!it.get.hasNext,
            cb.goto(LendOfStream))
          cb.assign(rv, Code.longValue(it.get.next()))
          cb.assign(rowIdx, rowIdx + 1L)


          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb => IEmitCode.present(cb,
          if (insertUID) {
            val uid = EmitValue.present(
              new SStackStructValue(uidSType, Array(
                EmitValue.present(primitive(partIdx)),
                EmitValue.present(primitive(rowIdx)))))
            eltPType.loadCheapSCode(cb, rv)._insert(requestedType, uidFieldName -> uid)
          } else {
            eltPType.loadCheapSCode(cb, rv)
          }))

        override def close(cb: EmitCodeBuilder): Unit = {}
      }

      SStreamValue(producer)
    }
  }

  def toJValue: JValue = {
    JObject(
      "category" -> JString("PartitionIteratorLongReader"),
      "fullRowType" -> Extraction.decompose(fullRowType)(PartitionReader.formats),
      "uidFieldName" -> JString(uidFieldName),
      "contextType" -> Extraction.decompose(contextType)(PartitionReader.formats))
  }
}

class GenericTableValueRDDPartition(
  val index: Int,
  val context: Any
) extends Partition

class GenericTableValueRDD(
  @transient val contexts: IndexedSeq[Any],
  body: (Region, HailClassLoader, Any) => Iterator[Long]
) extends RDD[RVDContext => Iterator[Long]](SparkBackend.sparkContext("GenericTableValueRDD"), Nil) {
  def getPartitions: Array[Partition] = contexts.zipWithIndex.map { case (c, i) =>
    new GenericTableValueRDDPartition(i, c)
  }.toArray

  def compute(split: Partition, context: TaskContext): Iterator[RVDContext => Iterator[Long]] = {
    Iterator.single { (rvdCtx: RVDContext) =>
      body(rvdCtx.region, theHailClassLoaderForSparkWorkers, split.asInstanceOf[GenericTableValueRDDPartition].context)
    }
  }
}

abstract class LoweredTableReaderCoercer {
  def coerce[M[_]: MonadLower](
    ctx: ExecuteContext,
    globals: IR,
    contextType: Type,
    contexts: IndexedSeq[Any],
    body: IR => IR
  ): M[TableStage]
}

class GenericTableValue(
  val fullTableType: TableType,
  val uidFieldName: String,
  val partitioner: Option[RVDPartitioner],
  val globals: TStruct => Row,
  val contextType: TStruct,
  var contexts: IndexedSeq[Any],
  val bodyPType: TStruct => PStruct,
  val body: TStruct => (Region, HailClassLoader, FS, Any) => Iterator[Long]) {

  assert(fullTableType.rowType.hasField(uidFieldName), s"uid=$uidFieldName, t=$fullTableType")
  assert(contextType.hasField("partitionIndex"))
  assert(contextType.fieldType("partitionIndex") == TInt32)

  var ltrCoercer: LoweredTableReaderCoercer = _
  def getLTVCoercer[M[_]](ctx: ExecuteContext, context: String, cacheKey: Any)
                         (implicit M: MonadLower[M])
  : M[LoweredTableReaderCoercer] = {
    if (ltrCoercer != null) M.pure(ltrCoercer)
    else LoweredTableReader.makeCoercer(
        ctx,
        fullTableType.key,
        1,
        uidFieldName,
        contextType,
        contexts,
        fullTableType.keyType,
        bodyPType,
        body,
        context,
        cacheKey
      ).map { c =>
      ltrCoercer = c
      c
    }
  }

  def toTableStage[M[_]](ctx: ExecuteContext, requestedType: TableType, context: String, cacheKey: Any)
                        (implicit M: MonadLower[M]): M[TableStage] = {
    val globalsIR = Literal(requestedType.globalType, globals(requestedType.globalType))
    val requestedBody: (IR) => (IR) = (ctx: IR) => ReadPartition(ctx,
      requestedType.rowType,
      new PartitionIteratorLongReader(
        fullTableType.rowType, uidFieldName, contextType,
        (requestedType: Type) => bodyPType(requestedType.asInstanceOf[TStruct]),
        (requestedType: Type) => body(requestedType.asInstanceOf[TStruct])))
    var p: RVDPartitioner = null
    partitioner match {
      case Some(partitioner) =>
        p = partitioner
      case None if requestedType.key.isEmpty =>
        p = RVDPartitioner.unkeyed(ctx.stateManager, contexts.length)
      case None =>
    }
    if (p != null)
      M.pure {
        TableStage(globalsIR, p, TableStageDependency.none, ToStream(Literal(TArray(contextType), contexts)), requestedBody)
      }
    else
      getLTVCoercer(ctx, context, cacheKey).flatMap(
        _.coerce(ctx,globalsIR,contextType, contexts,requestedBody)
      )
  }

  def toContextRDD(fs: FS, requestedRowType: TStruct): ContextRDD[Long] = {
    val localBody = body(requestedRowType)
    ContextRDD(new GenericTableValueRDD(contexts, localBody(_, _, fs, _)))
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
          RVDPartitioner.unkeyed(ctx.stateManager, contexts.length),
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
