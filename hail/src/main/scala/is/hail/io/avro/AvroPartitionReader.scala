package is.hail.io.avro

import is.hail.annotations.Region
import is.hail.asm4s.{Field => _, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.streams.StreamProducer
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitValue, IEmitCode, PartitionReader}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SStreamValue, primitive}
import is.hail.types.physical.{PCanonicalTuple, PInt64Required}
import is.hail.types.virtual._
import is.hail.types.{RField, RStruct, TypeWithRequiredness}
import org.apache.avro.Schema
import org.apache.avro.file.DataFileStream
import org.apache.avro.generic.{GenericData, GenericDatumReader, GenericRecord}
import org.apache.avro.io.DatumReader
import org.json4s.{Extraction, JValue}

import java.io.InputStream
import scala.collection.JavaConverters._

case class AvroPartitionReader(schema: Schema, uidFieldName: String) extends PartitionReader {
  def contextType: Type = TStruct("partitionPath" -> TString, "partitionIndex" -> TInt64)

  def fullRowTypeWithoutUIDs: TStruct = AvroReader.schemaToType(schema)

  lazy val fullRowType: TStruct = fullRowTypeWithoutUIDs
    .appendKey(uidFieldName, TTuple(TInt64, TInt64))

  override def rowRequiredness(requestedType: TStruct): RStruct = {
    val req = TypeWithRequiredness.apply(requestedType).asInstanceOf[RStruct]
    val concreteFields = if (requestedType.hasField(uidFieldName))
      req.fields.init
    else
      req.fields

    concreteFields.foreach { case RField(name, typ, _) =>
      AvroReader.setRequiredness(schema.getField(name).schema, typ)
    }

    if (requestedType.hasField(uidFieldName))
      req.fields.last.typ.fromPType(PCanonicalTuple(true, PInt64Required, PInt64Required))

    req.hardSetRequiredness(true)
    req
  }

  override def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct
  ): IEmitCode = {
    context.toI(cb).map(cb) { case ctxStruct: SBaseStructValue =>
      val partIdx = cb.memoizeField(ctxStruct.loadField(cb, "partitionIndex").get(cb), "partIdx")
      val pathString = ctxStruct.loadField(cb, "partitionPath").get(cb).asString.loadString(cb)

      val makeUID = requestedType.hasField(uidFieldName)
      val concreteRequestedType = if (makeUID)
        requestedType.deleteKey(uidFieldName)
      else
        requestedType

      val it = mb.genFieldThisRef[DataFileStream[GenericRecord]]("datafilestream")
      val record = mb.genFieldThisRef[GenericRecord]("record")
      val region = mb.genFieldThisRef[Region]("region")
      val rowIdx = mb.genFieldThisRef[Long]("rowIdx")

      val producer = new StreamProducer {
        override def method: EmitMethodBuilder[_] = cb.emb
        val length: Option[EmitCodeBuilder => Code[Int]] = None

        def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
          val mb = cb.emb
          val codeSchema = cb.newLocal("schema", mb.getObject(schema))
          cb.assign(record, Code.newInstance[GenericData.Record, Schema](codeSchema))
          val is = mb.open(pathString, false)
          val datumReader = Code.newInstance[GenericDatumReader[GenericRecord], Schema](codeSchema)
          val dataFileStream = Code.newInstance[DataFileStream[GenericRecord], InputStream, DatumReader[GenericRecord]](is, datumReader)

          cb.assign(it, dataFileStream)
          cb.assign(rowIdx, -1L)
        }

        val elementRegion: Settable[Region] = region
        val requiresMemoryManagementPerElement: Boolean = true
        val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.if_(!it.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(record, it.invoke[AnyRef, GenericRecord]("next", record))
          cb.assign(rowIdx, rowIdx + 1L)
          cb.goto(LproduceElementDone)
        }

        val element: EmitCode = EmitCodeBuilder.scopedEmitCode(mb) { cb =>
          val baseStruct = AvroReader.recordToHail(cb, region, record, concreteRequestedType)
          if (makeUID) {
            val uid = EmitValue.present(
              SStackStruct.constructFromArgs(cb, region, TTuple(TInt64, TInt64),
                EmitCode.present(mb, partIdx),
                EmitCode.present(mb, primitive(rowIdx))))
            EmitCode.present(mb, baseStruct._insert(requestedType, uidFieldName -> uid))
          } else {
            EmitCode.present(mb, baseStruct)
          }
        }

        def close(cb: EmitCodeBuilder): Unit = cb += it.invoke[Unit]("close")
      }

      SStreamValue(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

object AvroReader {
  private[avro] def schemaToType(schema: Schema): TStruct = {
    try {
      if (schema.getType != Schema.Type.RECORD) {
        throw new UnsupportedOperationException("hail conversion from avro is only supported for top level record types")
      }
      _schemaToType(schema).asInstanceOf[TStruct]
    } catch {
      case e: UnsupportedOperationException => throw new UnsupportedOperationException(s"hail conversion from $schema is unsupported", e)
    }
  }

  private def _schemaToType(schema: Schema): Type = schema.getType match {
    case Schema.Type.BOOLEAN => TBoolean
    case Schema.Type.INT => TInt32
    case Schema.Type.LONG => TInt64
    case Schema.Type.FLOAT => TFloat32
    case Schema.Type.DOUBLE => TFloat64

    case Schema.Type.STRING => TString
    case Schema.Type.BYTES => TBinary
    case Schema.Type.RECORD =>
      TStruct(schema.getFields.asScala.map(f => (f.name(), _schemaToType(f.schema()))): _*)
    case Schema.Type.UNION =>
      val types = schema.getTypes
      // we only support ["null", type] (or [type, "null"]) for unions as nullable data
      val nullIndex = types.asScala.indexWhere(s => s.getType == Schema.Type.NULL)
      if (types.size() != 2 || nullIndex == -1)
        throw new UnsupportedOperationException(s"hail conversion from avro $schema is unsupported")
      _schemaToType(types.get(1 - nullIndex))

    case _ => throw new UnsupportedOperationException(s"hail conversion from avro $schema is unsupported")
  }

  private[avro] def recordToHail(cb: EmitCodeBuilder, region: Value[Region], record: Value[GenericRecord], requestedType: TBaseStruct): SBaseStructValue = {
    val codes = requestedType.fields.map { case Field(name, typ, _) =>
      val v = cb.newLocal[AnyRef]("avro_value")
      cb.assign(v, record.invoke[String, AnyRef]("get", name))
      typ match {
        case TBoolean =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, primitive(cb.memoize(Code.booleanValue(Code.checkcast[java.lang.Boolean](v))))))
        case TInt32 =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, primitive(cb.memoize(Code.intValue(Code.checkcast[java.lang.Number](v))))))
        case TInt64 =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, primitive(cb.memoize(Code.longValue(Code.checkcast[java.lang.Number](v))))))
        case TFloat32 =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, primitive(cb.memoize(Code.floatValue(Code.checkcast[java.lang.Number](v))))))
        case TFloat64 =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, primitive(cb.memoize(Code.doubleValue(Code.checkcast[java.lang.Number](v))))))
        case TString =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, new SJavaStringValue(cb.memoize(Code.checkcast[org.apache.avro.util.Utf8](v).invoke[String]("toString")))))
        case TBinary =>
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, new SJavaBytesValue(cb.memoize(Code.checkcast[Array[Byte]](v)))))
        case typ: TBaseStruct =>
          val record = cb.newLocal[GenericRecord]("avro_subrecord", Code.checkcast[GenericRecord](v))
          EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, v.isNull, recordToHail(cb, region, record, typ)))
      }
    }

    SStackStruct.constructFromArgs(cb, region, requestedType, codes: _*)
  }

  private[avro] def setRequiredness(schema: Schema, typ: TypeWithRequiredness): Unit = {
    val realSchema = schema.getType match {
      case Schema.Type.UNION =>
        typ.hardSetRequiredness(false)
        val nullIndex = schema.getTypes.asScala.indexWhere(s => s.getType == Schema.Type.NULL)
        schema.getTypes.get(1 - nullIndex)
      case _ =>
        typ.hardSetRequiredness(true)
        schema
    }

    typ match {
      case t: RStruct => t.fields.foreach { case RField(name, typ, _) =>
        setRequiredness(realSchema.getField(name).schema, typ)
      }
      case _ => // do nothing
    }
  }
}
