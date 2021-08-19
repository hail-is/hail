package is.hail.io.avro

import java.io.InputStream

import is.hail.annotations.Region
import is.hail.asm4s.{Code, CodeLabel, Settable, Value}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, ExecuteContext, IEmitCode, PartitionReader}
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical.stypes.interfaces.SStreamCode
import is.hail.types.virtual._
import org.json4s.{Extraction, JValue}
import org.apache.avro.Schema
import org.apache.avro.file.DataFileStream
import org.apache.avro.generic.{GenericData, GenericDatumReader, GenericRecord}

import scala.collection.JavaConverters._

class AvroPartitionReader(schema: Schema) extends PartitionReader {
  def contextType: Type = TString

  lazy val fullRowType: Type = AvroReader.schemaToType(schema)

  def rowRequiredness(requestedType: Type): TypeWithRequiredness = ???

  def emitStream(ctx: ExecuteContext, cb: EmitCodeBuilder, context: EmitCode, partitionRegion: Value[Region], requestedType: Type): IEmitCode = {
    context.toI(cb).map(cb) { path =>
      val mb = cb.emb
      val it = mb.genFieldThisRef[DataFileStream[GenericRecord]]("datafilestream")
      val record = mb.genFieldThisRef[GenericRecord]("record")
      val region = mb.genFieldThisRef[Region]("region")

      val producer = new StreamProducer {
        val length: Option[EmitCodeBuilder => Code[Int]] = None

        def initialize(cb: EmitCodeBuilder): Unit = {
          val mb = cb.emb
          cb.assign(region, partitionRegion)
          val codeSchema = cb.newLocal("schema", mb.getObject(schema))
          cb.assign(record, Code.newInstance[GenericData.Record, Schema](codeSchema))
          val is = mb.open(path.asString.loadString(), false)
          val datumReader = Code.newInstance[GenericDatumReader[GenericRecord], Schema](codeSchema)
          val dataFileStream = Code.newInstance[DataFileStream[GenericRecord], InputStream, GenericDatumReader[GenericRecord]](is, datumReader)

          cb.assign(it, dataFileStream)
        }

        val elementRegion: Settable[Region] = region
        val requiresMemoryManagementPerElement: Boolean = false
        val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          cb.ifx(!it.invoke[Boolean]("hasNext"), cb.goto(LendOfStream))
          cb.assign(record, it.invoke[GenericRecord, GenericRecord]("next", record))
          cb.goto(LproduceElementDone)
        }

        val element: EmitCode = ??? // need to read all the record fields (recursively) and return a stack struct
                                    // where all the codes are the codes necessary for grabbing the various
                                    // fields

        def close(cb: EmitCodeBuilder): Unit = cb += it.invoke("close")
      }

      SStreamCode(producer)
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}

object AvroReader {
  def schemaToType(schema: Schema): Type = {
    try {
      if (schema.getType != Schema.Type.RECORD) {
        throw new UnsupportedOperationException("hail conversion from avro is only supported for top level record types")
      }
      _schemaToType(schema)
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
      val types = schema.getTypes()
      // we only support ["null", type] for unions as nullable data
      if (!(types.size() == 2 && types.get(0).getType == Schema.Type.NULL))
        throw new UnsupportedOperationException(s"hail conversion from avro $schema is unsupported")
      _schemaToType(types.get(1))

    case _ => throw new UnsupportedOperationException(s"hail conversion from avro $schema is unsupported")
  }
}