package is.hail.expr.ir

import is.hail.annotations.{BroadcastRow, Region}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir.functions.UtilFunctions
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.types.{RStruct, TableType, TypeWithRequiredness}
import is.hail.types.physical.{PStruct, PType}
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SStackStruct, SStackStructValue}
import is.hail.types.physical.stypes.interfaces.{primitive, SBaseStructValue, SStream, SStreamValue}
import is.hail.types.physical.stypes.primitives.{SInt64, SInt64Value}
import is.hail.types.virtual.{TArray, TInt32, TInt64, TStruct, TTuple, Type}
import is.hail.utils._

import org.json4s.{Extraction, JValue}
import org.json4s.JsonAST.{JObject, JString}

import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

class PartitionIteratorLongReader(
  rowType: TStruct,
  override val uidFieldName: String,
  override val contextType: TStruct,
  bodyPType: TStruct => PStruct,
  body: TStruct => (Region, HailClassLoader, FS, Any) => Iterator[Long],
) extends PartitionReader {
  assert(contextType.hasField("partitionIndex"))
  assert(contextType.fieldType("partitionIndex") == TInt32)

  override lazy val fullRowType: TStruct = if (rowType.hasField(uidFieldName)) {
    rowType
  } else {
    rowType.insertFields(Array(uidFieldName -> TTuple(TInt64, TInt64)))
  }

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
    requestedType: TStruct,
  ): IEmitCode = {

    val insertUID: Boolean = requestedType.hasField(uidFieldName) && !rowType.hasField(uidFieldName)

    val concreteType: TStruct = if (insertUID)
      requestedType.deleteKey(uidFieldName)
    else
      requestedType

    val eltPType = bodyPType(concreteType)
    val uidSType: SStackStruct = SStackStruct(
      TTuple(TInt64, TInt64),
      Array(EmitType(SInt64, true), EmitType(SInt64, true)),
    )

    context.toI(cb).map(cb) { case _ctxStruct: SBaseStructValue =>
      val ctxStruct = cb.memoizeField(_ctxStruct, "ctxStruct")
      val partIdx =
        cb.memoizeField(_ctxStruct.loadField(cb, "partitionIndex").get(cb).asInt.value.toL)
      val rowIdx = mb.genFieldThisRef[Long]("pnr_rowidx")
      val region = mb.genFieldThisRef[Region]("pilr_region")
      val it = mb.genFieldThisRef[Iterator[java.lang.Long]]("pilr_it")
      val rv = mb.genFieldThisRef[Long]("pilr_rv")

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb
        override val length: Option[EmitCodeBuilder => Code[Int]] = None

        override def initialize(cb: EmitCodeBuilder, partitionRegion: Value[Region]): Unit = {
          val ctxJavaValue = UtilFunctions.svalueToJavaValue(cb, partitionRegion, ctxStruct)
          cb.assign(
            it,
            cb.emb.getObject(body(requestedType))
              .invoke[
                java.lang.Object,
                java.lang.Object,
                java.lang.Object,
                java.lang.Object,
                Iterator[java.lang.Long],
              ](
                "apply",
                region,
                cb.emb.getHailClassLoader,
                cb.emb.getFS,
                ctxJavaValue,
              ),
          )
        }

        override val elementRegion: Settable[Region] = region
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(!it.get.hasNext, cb.goto(LendOfStream))
          cb.assign(rv, Code.longValue(it.get.next()))
          cb.assign(rowIdx, rowIdx + 1L)

          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = EmitCode.fromI(mb)(cb =>
          IEmitCode.present(
            cb,
            if (insertUID) {
              val uid = EmitValue.present(
                new SStackStructValue(
                  uidSType,
                  Array(
                    EmitValue.present(primitive(partIdx)),
                    EmitValue.present(primitive(rowIdx)),
                  ),
                )
              )
              eltPType.loadCheapSCode(cb, rv)._insert(requestedType, uidFieldName -> uid)
            } else {
              eltPType.loadCheapSCode(cb, rv)
            },
          )
        )

        override def close(cb: EmitCodeBuilder): Unit = {}
      }

      SStreamValue(producer)
    }
  }

  def toJValue: JValue =
    JObject(
      "category" -> JString("PartitionIteratorLongReader"),
      "fullRowType" -> Extraction.decompose(fullRowType)(PartitionReader.formats),
      "uidFieldName" -> JString(uidFieldName),
      "contextType" -> Extraction.decompose(contextType)(PartitionReader.formats),
    )
}

abstract class LoweredTableReaderCoercer {
  def coerce(
    ctx: ExecuteContext,
    globals: IR,
    contextType: Type,
    contexts: IndexedSeq[Any],
    body: IR => IR,
  ): TableStage
}

class GenericTableValue(
  val fullTableType: TableType,
  val uidFieldName: String,
  val partitioner: Option[RVDPartitioner],
  val globals: TStruct => Row,
  val contextType: TStruct,
  var contexts: IndexedSeq[Any],
  val bodyPType: TStruct => PStruct,
  val body: TStruct => (Region, HailClassLoader, FS, Any) => Iterator[Long],
) {

  assert(fullTableType.rowType.hasField(uidFieldName), s"uid=$uidFieldName, t=$fullTableType")
  assert(contextType.hasField("partitionIndex"))
  assert(contextType.fieldType("partitionIndex") == TInt32)

  private var ltrCoercer: LoweredTableReaderCoercer = _

  private def getLTVCoercer(ctx: ExecuteContext, context: String, cacheKey: Any)
    : LoweredTableReaderCoercer = {
    if (ltrCoercer == null) {
      ltrCoercer = LoweredTableReader.makeCoercer(
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
        cacheKey,
      )
    }
    ltrCoercer
  }

  def toTableStage(ctx: ExecuteContext, requestedType: TableType, context: String, cacheKey: Any)
    : TableStage = {
    val globalsIR = Literal(requestedType.globalType, globals(requestedType.globalType))
    val requestedBody: (IR) => (IR) = (ctx: IR) =>
      ReadPartition(
        ctx,
        requestedType.rowType,
        new PartitionIteratorLongReader(
          fullTableType.rowType,
          uidFieldName,
          contextType,
          (requestedType: Type) => bodyPType(requestedType.asInstanceOf[TStruct]),
          (requestedType: Type) => body(requestedType.asInstanceOf[TStruct]),
        ),
      )
    var p: RVDPartitioner = null
    partitioner match {
      case Some(partitioner) =>
        p = partitioner
      case None if requestedType.key.isEmpty =>
        p = RVDPartitioner.unkeyed(ctx.stateManager, contexts.length)
      case None =>
    }
    if (p != null) {
      val contextsIR = ToStream(Literal(TArray(contextType), contexts))
      TableStage(globalsIR, p, TableStageDependency.none, contextsIR, requestedBody)
    } else {
      getLTVCoercer(ctx, context, cacheKey).coerce(
        ctx,
        globalsIR,
        contextType, contexts,
        requestedBody)
    }
  }
}
