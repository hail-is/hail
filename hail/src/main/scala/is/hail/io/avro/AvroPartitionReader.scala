package is.hail.io.avro

import java.io.InputStream

import is.hail.annotations.Region
import is.hail.asm4s.{Code, CodeLabel, Settable, Value}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, ExecuteContext, IEmitCode, PartitionReader}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SStreamCode, primitive}
import is.hail.types.virtual._
import is.hail.types.{RField, RStruct, TypeWithRequiredness}
import org.apache.avro.Schema
import org.apache.avro.io.DatumReader
import org.apache.avro.file.DataFileStream
import org.apache.avro.generic.{GenericData, GenericDatumReader, GenericRecord}
import org.json4s.{Extraction, JValue}

import scala.collection.JavaConverters._

case class AvroPartitionReader(schema: Schema) extends PartitionReader {
  def contextType: Type = TString

  val fullRowType: TStruct = AvroReader.schemaToType(schema)

  def rowRequiredness(requestedType: Type): TypeWithRequiredness = {
    val req = TypeWithRequiredness.apply(requestedType).asInstanceOf[RStruct]
    req.fields.foreach { case RField(name, typ, _) =>
      AvroReader.setRequiredness(schema.getField(name).schema, typ)
    }
    req.hardSetRequiredness(true)
    req
  }

  def emitStream(ctx: ExecuteContext, cb: EmitCodeBuilder, context: EmitCode, partitionRegion: Value[Region], requestedType: Type): IEmitCode = {
    val structType = requestedType.asInstanceOf[TStruct]
    context.toI(cb).map(cb) { path =>
      val mb = cb.emb
      val it = mb.genFieldThisRef[DataFileStream[GenericRecord]]("datafilestream")
      val record = mb.genFieldThisRef[GenericRecord]("record")
      val region = mb.genFieldThisRef[Region]("region")

      val producer = new StreamProducer {
        val length: Option[EmitCodeBuilder => Code[Int]] = None

        def initialize(cb: EmitCodeBuilder): Unit = {
          val mb = cb.emb
          val codeSchema = cb.newLocal("schema", mb.getObject(schema))
          cb.assign(record, Code.newInstance[GenericData.Record, Schema](codeSchema))
          val is = mb.open(path.asString.loadString(), false)
          val datumReader = Code.newInstance[GenericDatumReader[GenericRecord], Schema](codeSchema)
          val dataFileStream = Code.newInstance[DataFileStream[GenericRecord], InputStream, DatumReader[GenericRecord]](is, datumReader)

          cb.assign(it, dataFileStream)
        }

        val elementRegion: Settable[Region] = region
        val requiresMemoryManagementPerElement: Boolean = true
        val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!it.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(record, it.invoke[AnyRef, GenericRecord]("next", record))
          cb.goto(LproduceElementDone)
        }

        val element: EmitCode = EmitCodeBuilder.scopedEmitCode(mb) { cb =>
          EmitCode.present(mb, AvroReader.recordToHail(cb, region, record, structType))
        }

        def close(cb: EmitCodeBuilder): Unit = cb += it.invoke[Unit]("close")
      }

      SStreamCode(producer)
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

  private[avro] def recordToHail(cb: EmitCodeBuilder, region: Value[Region], record: Value[GenericRecord], requestedType: TBaseStruct): SBaseStructCode = {
    val codes = requestedType.fields.map { case Field(name, typ, _) =>
      val v = cb.newLocal[AnyRef]("avro_value")
      cb.assign(v, record.invoke[String, AnyRef]("get", name))
      typ match {
        case TBoolean =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, primitive(Code.booleanValue(Code.checkcast[java.lang.Boolean](v)))))
        case TInt32 =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, primitive(Code.intValue(Code.checkcast[java.lang.Number](v)))))
        case TInt64 =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, primitive(Code.longValue(Code.checkcast[java.lang.Number](v)))))
        case TFloat32 =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, primitive(Code.floatValue(Code.checkcast[java.lang.Number](v)))))
        case TFloat64 =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, primitive(Code.doubleValue(Code.checkcast[java.lang.Number](v)))))
        case TString =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, new SJavaStringCode(Code.checkcast[org.apache.avro.util.Utf8](v).invoke[String]("toString"))))
        case TBinary =>
          EmitCode.fromI(cb.emb)(IEmitCode(_, v.isNull, new SJavaBytesCode(Code.checkcast[Array[Byte]](v))))
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
