package is.hail.utils

import java.io.{ObjectInputStream, ObjectOutputStream, Serializable}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Output, Input}

import org.apache.hadoop

class SerializableHadoopConfiguration(@volatile @transient var value: hadoop.conf.Configuration) extends Serializable with KryoSerializable {
  private def writeObject(out: ObjectOutputStream) {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream) {
    value = new hadoop.conf.Configuration(false)
    value.readFields(in)
  }

  override def write(kryo: Kryo, output: Output) {
    value.write(new ObjectOutputStream(output.getOutputStream()))
  }

  override def read(kryo: Kryo, input: Input) {
    value = new hadoop.conf.Configuration(false)
    value.readFields(new ObjectInputStream(input.getInputStream()))
  }

}
