package is.hail.annotations

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr.types.Type
import is.hail.io.{CodecSpec, Decoder, Encoder}
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object RegionValue {
  def apply(): RegionValue = new RegionValue(null, 0)

  def apply(region: Region): RegionValue = new RegionValue(region, 0)

  def apply(region: Region, offset: Long) = new RegionValue(region, offset)
}

final class RegionValue(var region: Region,
  var offset: Long) {
  def set(newRegion: Region, newOffset: Long) {
    region = newRegion
    offset = newOffset
  }

  def setRegion(newRegion: Region) {
    region = newRegion
  }

  def setOffset(newOffset: Long) {
    offset = newOffset
  }

  def pretty(t: Type): String = region.pretty(t, offset)

  def copy(): RegionValue = RegionValue(region.copy(), offset)


  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}

object SerializedRegionValue {

  def apply(t: Type, region: Region, offset: Long, baos: ByteArrayOutputStream, enc: Encoder): SerializedRegionValue = {
    baos.reset()
    enc.writeRegionValue(t, region, offset)
    enc.flush()
    new SerializedRegionValue(baos.toByteArray)
  }

  def apply(t: Type, rv: RegionValue, baos: ByteArrayOutputStream, enc: Encoder): SerializedRegionValue = {
    apply(t, rv.region, rv.offset, baos, enc)
  }

  def getEncoder(codec: CodecSpec = CodecSpec.default): (ByteArrayOutputStream, Encoder) = {
    val baos = new ByteArrayOutputStream()
    val enc = codec.buildEncoder(baos)
    (baos, enc)
  }

  def getRVSerializer(t: Type, codec: CodecSpec = CodecSpec.default): RegionValue => SerializedRegionValue = {
    val (baos, enc) = getEncoder(codec)
    val serializer = { rv: RegionValue => apply(t, rv, baos, enc) }
    serializer
  }
}

class SerializedRegionValue(var value: Array[Byte]) extends KryoSerializable with Serializable {

  lazy val codec: CodecSpec = CodecSpec.default
  lazy val bais: ByteArrayInputStream = new ByteArrayInputStream(value)
  lazy val dec: Decoder = codec.buildDecoder(bais)

  def deserialize(t: Type, region: Region): Long = {
    bais.reset()
    dec.readRegionValue(t, region)
  }

  def toRegionValue(t: Type, region: Region): RegionValue = {
    RegionValue(region, deserialize(t, region))
  }

  def pretty(t: Type): String = {
    val region = Region()
    region.pretty(t, deserialize(t, region))
  }

  override def write(kryo: Kryo, output: Output) {
    output.writeInt(value.length)
    output.write(value, 0, value.length)
  }

  override def read(kryo: Kryo, input: Input) {
    val end = input.readInt()
    value = new Array[Byte](end)
    input.read(value)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeInt(value.length)
    out.write(value, 0, value.length)
  }

  private def readObject(in: ObjectInputStream) {
    val end = in.readInt()
    value = new Array[Byte](end)
    in.read(value)
  }
}