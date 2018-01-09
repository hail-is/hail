package is.hail.kryo

import org.apache.spark.serializer.KryoRegistrator
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.serializers.JavaSerializer
import is.hail.annotations.{Region, UnsafeIndexedSeq, UnsafeRow}
import is.hail.utils.{Interval, SerializableHadoopConfiguration}

class HailKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[SerializableHadoopConfiguration], new JavaSerializer())
    kryo.register(classOf[UnsafeRow])
    kryo.register(classOf[UnsafeIndexedSeq])
    kryo.register(classOf[Region])

    // work around https://github.com/EsotericSoftware/kryo/pull/342
    // which is fixed in 4.0.0 but Spark ships with (shaded) 3.0.3
    kryo.register(classOf[Interval[_]], new JavaSerializer())
  }
}
