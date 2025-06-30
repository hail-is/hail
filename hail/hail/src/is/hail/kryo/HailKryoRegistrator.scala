package is.hail.kryo

import is.hail.annotations.{Region, UnsafeIndexedSeq, UnsafeRow}
import is.hail.macros.void
import is.hail.utils.{Interval, SerializableHadoopConfiguration}
import is.hail.variant.Locus

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.serializers.JavaSerializer
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.sql.catalyst.expressions.GenericRow

class HailKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo): Unit = {
    void(kryo.register(classOf[SerializableHadoopConfiguration], new JavaSerializer()))
    void(kryo.register(classOf[UnsafeRow]))
    void(kryo.register(classOf[GenericRow]))
    void(kryo.register(classOf[Locus]))
    void(kryo.register(classOf[Interval]))
    void(kryo.register(classOf[UnsafeIndexedSeq]))
    void(kryo.register(classOf[Region]))
  }
}
