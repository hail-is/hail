package is.hail.kryo

import is.hail.annotations.{Region, RegionMemory, RegionPool, RowSeq, UnsafeIndexedSeq, UnsafeRow}
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichArray
import is.hail.utils.{Interval, IntervalEndpoint, SerializableHadoopConfiguration}
import is.hail.variant.Locus

import com.esotericsoftware.kryo.{Kryo, Serializer}
import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.serializers.JavaSerializer
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.sql.catalyst.expressions.GenericRow

class HailKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo): Unit = {
    kryo.addDefaultSerializer(classOf[ArraySeq[_]], new ArraySeqSerializer)
    kryo.register(classOf[SerializableHadoopConfiguration], new JavaSerializer())
    kryo.register(classOf[UnsafeRow])
    kryo.register(classOf[GenericRow])
    kryo.register(classOf[RowSeq])
    kryo.register(classOf[Locus])
    kryo.register(classOf[Interval])
    kryo.register(classOf[IntervalEndpoint])
    kryo.register(classOf[UnsafeIndexedSeq])
    kryo.register(classOf[Region])
    kryo.register(classOf[RegionPool])
    kryo.register(classOf[RegionMemory])
  }
}

// Recall that primitive and object arrays are not subtypes on the jvm.
// We write the underlying array type to ensure that we recover the correct
// array type from the input stream.
class ArraySeqSerializer extends Serializer[ArraySeq[_]](true, true) {
  override def write(kryo: Kryo, output: Output, xs: ArraySeq[_]): Unit =
    kryo.writeClassAndObject(output, xs.unsafeArray)

  override def read(kryo: Kryo, input: Input, `type`: Class[ArraySeq[_]]): ArraySeq[_] =
    kryo.readClassAndObject(input).asInstanceOf[Array[_]].unsafeToArraySeq
}
