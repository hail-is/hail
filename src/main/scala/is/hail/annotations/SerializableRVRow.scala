package is.hail.annotations

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.utils._

object SerializableRVRow {

  def apply(ur: UnsafeRow): SerializableRVRow = {
    val aos = new ByteArrayOutputStream()
    val enc = CodecSpec.default.buildEncoder(aos)
    enc.writeRegionValue(ur.t, ur.region, ur.offset)
    enc.flush()

    new SerializableRVRow(ur.t, aos.toByteArray)
  }
}

class SerializableRVRow(var t: TBaseStruct, var bytes: Array[Byte]) extends KryoSerializable {

  override def write(kryo: Kryo, output: Output) {
    output.writeBoolean(t.isInstanceOf[TStruct])
    kryo.writeObject(output, t)

    output.writeInt(bytes.length)
    output.write(bytes, 0, bytes.length)
  }

  override def read(kryo: Kryo, input: Input) {
    val isStruct = input.readBoolean()
    t = kryo.readObject(input, if (isStruct) classOf[TStruct] else classOf[TTuple])

    val smallInOff = input.readInt()
    bytes = new Array[Byte](smallInOff)
    input.readFully(bytes, 0, smallInOff)

  }

  def toRegion(region: Region): Long = {
    var offset = 0L
    using(CodecSpec.default.buildDecoder(new ByteArrayInputStream(bytes))) { dec =>
      offset = dec.readRegionValue(t, region)
    }
    offset
  }
}
