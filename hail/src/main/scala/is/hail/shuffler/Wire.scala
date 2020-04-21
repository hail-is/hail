package is.hail.shuffler

import java.io._

import com.fasterxml.jackson.core.{JsonGenerator, JsonParser}
import is.hail.expr.ir.IRParser
import is.hail.expr.types.virtual.TStruct
import is.hail.io.TypedCodecSpec
import is.hail.rvd.AbstractRVDSpec
import org.json4s.jackson.{JsonMethods, Serialization}

object Wire {
  val EOS: Byte = -1.toByte
  val START: Byte = 0.toByte
  val PUT: Byte = 1.toByte
  val GET: Byte = 2.toByte
  val STOP: Byte = 3.toByte

  val ID_SIZE = 4096 / 8

  import AbstractRVDSpec.formats

  // Jackson closes the InputStreams and OutputStreams you pass to the
  // ObjectMapper unless you set these flags, respectively, to false
  JsonMethods.mapper.configure(JsonParser.Feature.AUTO_CLOSE_SOURCE, false);
  JsonMethods.mapper.configure(JsonGenerator.Feature.AUTO_CLOSE_TARGET, false);

  def writeTStruct(out: DataOutputStream, x: TStruct): Unit = {
    out.writeUTF(x.parsableString())
  }

  def readTStruct(in: DataInputStream): TStruct = {
    IRParser.parseStructType(in.readUTF())
  }

  def writeTypedCodecSpec(out: DataOutputStream, x: TypedCodecSpec): Unit = {
    Serialization.write(x, out)
  }

  def readTypedCodecSpec(in: DataInputStream): TypedCodecSpec = {
    JsonMethods.parse(in).extract[TypedCodecSpec]
  }

  def writeStringArray(out: DataOutputStream, x: Array[String]): Unit = {
    val n = x.length
    out.writeInt(n)
    var i = 0
    while (i < n) {
      out.writeUTF(x(i))
      i += 1
    }
  }

  def readStringArray(in: DataInputStream): Array[String] = {
    val n = in.readInt()
    val a = new Array[String](n)
    var i = 0
    while (i < n) {
      a(i) = in.readUTF()
      i += 1
    }
    a
  }

  def writeByteArray(out: DataOutputStream, x: Array[Byte]): Unit = {
    val n = x.length
    out.writeInt(n)
    out.write(x)
  }

  def readByteArray(in: DataInputStream): Array[Byte] = {
    val n = in.readInt()
    val a = new Array[Byte](n)
    in.read(a)
    a
  }
}

