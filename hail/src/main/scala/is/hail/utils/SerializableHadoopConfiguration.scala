package is.hail.utils

import java.io.{ObjectInputStream, ObjectOutputStream, Serializable}

import org.apache.hadoop

class SerializableHadoopConfiguration(@transient var value: hadoop.conf.Configuration)
    extends Serializable {
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    value.write(out)
  }

  private def readObject(in: ObjectInputStream): Unit = {
    value = new hadoop.conf.Configuration(false)
    value.readFields(in)
  }
}
