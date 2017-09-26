package is.hail.kryo

import org.apache.spark.serializer.KryoRegistrator
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.serializers.JavaSerializer
import is.hail.annotations.{MemoryBuffer, UnsafeIndexedSeq, UnsafeRow}
import is.hail.utils.SerializableHadoopConfiguration

class HailKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[SerializableHadoopConfiguration], new JavaSerializer())
    kryo.register(classOf[UnsafeRow])
    kryo.register(classOf[UnsafeIndexedSeq])
    kryo.register(classOf[MemoryBuffer])
  }
}
