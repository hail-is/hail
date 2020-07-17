package is.hail.services.shuffler

import java.util.Base64

import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.encoded._
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.io._
import org.json4s.jackson._
import com.fasterxml.jackson.core.{JsonGenerator, JsonParser}
import is.hail.expr.ir.IRParser
import is.hail.types.virtual.TStruct
import is.hail.io.TypedCodecSpec
import is.hail.rvd.AbstractRVDSpec
import org.json4s.jackson.{JsonMethods, Serialization}

object Wire {
  val EOS: Byte = 255.toByte
  val START: Byte = 0.toByte
  val PUT: Byte = 1.toByte
  val GET: Byte = 2.toByte
  val STOP: Byte = 3.toByte
  val PARTITION_BOUNDS: Byte = 4.toByte

  val ID_SIZE = 32

  // Jackson closes the InputStreams and OutputStreams you pass to the
  // ObjectMapper unless you set these flags, respectively, to false
  JsonMethods.mapper.configure(JsonParser.Feature.AUTO_CLOSE_SOURCE, false);
  JsonMethods.mapper.configure(JsonGenerator.Feature.AUTO_CLOSE_TARGET, false);

  def writeTStruct(out: OutputBuffer, x: TStruct): Unit = {
    out.writeUTF(x.parsableString())
  }

  def writeTStruct(out: Value[OutputBuffer], x: TStruct): Code[Unit] = {
    out.writeUTF(x.parsableString)
  }

  def serializeTStruct(x: TStruct): String = {
    x.parsableString()
  }

  def readTStruct(in: InputBuffer): TStruct = {
    IRParser.parseStructType(in.readUTF())
  }

  def deserializeTStruct(x: String): TStruct = {
    IRParser.parseStructType(x)
  }

  def serializePType(x: PType): String = {
    x.toString
  }

  def deserializePType(x: String): PType = {
    IRParser.parsePType(x)
  }

  def deserializePType(x: Code[String]): Code[PType] =
    Code.invokeScalaObject1[String, PType](
      Wire.getClass, "deserializePType", x)

  def writeEType(out: Value[OutputBuffer], x: EType): Code[Unit] = {
    out.writeUTF(x.parsableString())
  }

  def writeEType(out: OutputBuffer, x: EType): Unit = {
    out.writeUTF(x.parsableString())
  }

  def serializeEType(x: EType): String = {
    x.parsableString()
  }

  def readEType(in: InputBuffer): EType = {
    IRParser.parse(in.readUTF(), EType.eTypeParser)
  }

  def deserializeEType(x: String): EType = {
    IRParser.parse(x, EType.eTypeParser)
  }

  def writeEBaseStruct(out: Value[OutputBuffer], x: EBaseStruct): Code[Unit] = {
    out.writeUTF(x.parsableString())
  }

  def writeEBaseStruct(out: OutputBuffer, x: EBaseStruct): Unit = {
    out.writeUTF(x.parsableString())
  }

  def readEBaseStruct(in: InputBuffer): EBaseStruct = {
    IRParser.parse(in.readUTF(), EType.eTypeParser).asInstanceOf[EBaseStruct]
  }

  def writeStringArray(out: OutputBuffer, x: Array[String]): Unit = {
    val n = x.length
    out.writeInt(n)
    var i = 0
    while (i < n) {
      out.writeUTF(x(i))
      i += 1
    }
  }

  def writeStringArray(out: Value[OutputBuffer], x: Array[String]): Code[Unit] = Code(
    out.writeInt(x.length),
    Code(x.map(s => out.writeUTF(s))))

  def serializeStringArray(x: Array[String]): String = {
    val mb = new MemoryBuffer()
    val mob = new MemoryOutputBuffer(mb)
    writeStringArray(mob, x)
    mob.flush()
    Base64.getEncoder().encodeToString(mb.toByteArray())
  }

  def readStringArray(in: InputBuffer): Array[String] = {
    val n = in.readInt()
    val a = new Array[String](n)
    var i = 0
    while (i < n) {
      a(i) = in.readUTF()
      i += 1
    }
    a
  }

  def deserializeStringArray(x: String): Array[String] = {
    val bytes = Base64.getDecoder().decode(x)
    val mb = new MemoryBuffer()
    mb.set(bytes)
    readStringArray(new MemoryInputBuffer(mb))
  }

  def writeSortFieldArray(out: Value[OutputBuffer], x: IndexedSeq[SortField]): Code[Unit] = Code(
    out.writeInt(x.length),
    Code(x.map(sf => Code(
      out.writeUTF(sf.field),
      out.writeByte(sf.sortOrder.serialize)))))

  def writeSortFieldArray(out: OutputBuffer, x: IndexedSeq[SortField]): Unit = {
    out.writeInt(x.length)
    x.foreach { sf =>
      out.writeUTF(sf.field)
      out.writeByte(sf.sortOrder.serialize)
    }
  }

  def readSortFieldArray(in: InputBuffer): IndexedSeq[SortField] = {
    val n = in.readInt()
    val a = new Array[SortField](n)
    var i = 0
    while (i < n) {
      val field = in.readUTF()
      val sortOrder = SortOrder.deserialize(in.readByte())
      a(i) = SortField(field, sortOrder)
      i += 1
    }
    a
  }

  def writeByteArray(out: OutputBuffer, x: Array[Byte]): Unit = {
    val n = x.length
    out.writeInt(n)
    out.write(x)
  }

  def writeByteArray(out: Value[OutputBuffer], _x: Value[Array[Byte]]): Code[Unit] = {
    val x = new CodeArray(_x)
    Code(
      out.writeInt(x.length),
      out.write(_x.get))
  }

  def readByteArray(in: InputBuffer): Array[Byte] = {
    val n = in.readInt()
    in.readBytesArray(n)
  }

  def readByteArray(in: Value[InputBuffer]): Code[Array[Byte]] =
    in.readBytesArray(in.readInt())
}

