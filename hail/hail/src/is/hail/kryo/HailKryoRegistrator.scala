package is.hail.kryo

import is.hail.annotations.{Region, RowSeq, UnsafeIndexedSeq, UnsafeRow}
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.utils.{Interval, SerializableHadoopConfiguration}
import is.hail.variant.Locus

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.serializers.JavaSerializer
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.sql.catalyst.expressions.GenericRow

class HailKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo): Unit = {
    kryo.register(classOf[SerializableHadoopConfiguration], new JavaSerializer())
    kryo.register(classOf[UnsafeRow])
    kryo.register(classOf[GenericRow])
    kryo.register(classOf[RowSeq])
    kryo.register(classOf[ArraySeq.ofRef[_]])
    kryo.register(classOf[ArraySeq.ofByte])
    kryo.register(classOf[Locus])
    kryo.register(classOf[Interval])
    kryo.register(classOf[UnsafeIndexedSeq])
    kryo.register(classOf[Region])
  }
}
