package is.hail.utils.richUtils

import breeze.linalg.DenseMatrix
import is.hail.utils.{ArrayBuilder, JSONWriter, MultiArray2, Truncatable}
import is.hail.variant.Variant
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s.JValue

import scala.collection.{TraversableOnce, mutable}
import scala.language.implicitConversions
import scala.math.Ordering
import scala.reflect.ClassTag

trait Implicits {
  implicit def toRichAny(a: Any): RichAny = new RichAny(a)

  implicit def toRichArray[T](a: Array[T]): RichArray[T] = new RichArray(a)

  implicit def toRichByteArrayBuilder(t: ArrayBuilder[Byte]): RichByteArrayBuilder =
    new RichByteArrayBuilder(t)

  implicit def toRichBoolean(b: Boolean): RichBoolean = new RichBoolean(b)

  implicit def toRichDenseMatrixDouble(m: DenseMatrix[Double]): RichDenseMatrixDouble = new RichDenseMatrixDouble(m)

  implicit def toRichEnumeration[T <: Enumeration](e: T): RichEnumeration[T] = new RichEnumeration(e)

  implicit def toRichHadoopConfiguration(hConf: hadoop.conf.Configuration): RichHadoopConfiguration =
    new RichHadoopConfiguration(hConf)

  implicit def toRichIndexedRow(r: IndexedRow): RichIndexedRow = new RichIndexedRow(r)

  implicit def toRichIntPairTraversableOnce[V](t: TraversableOnce[(Int, V)]): RichIntPairTraversableOnce[V] =
    new RichIntPairTraversableOnce[V](t)

  implicit def toRichIterable[T](i: Iterable[T]): RichIterable[T] = new RichIterable(i)

  implicit def toRichIterable[T](a: Array[T]): RichIterable[T] = new RichIterable(a)

  implicit def toRichIterator[T](it: Iterator[T]): RichIterator[T] = new RichIterator[T](it)

  implicit def toRichIteratorOfByte(i: Iterator[Byte]): RichIteratorOfByte = new RichIteratorOfByte(i)

  implicit def toRichMap[K, V](m: Map[K, V]): RichMap[K, V] = new RichMap(m)

  implicit def toRichMultiArray2Long(ma: MultiArray2[Long]): RichMultiArray2Long = new RichMultiArray2Long(ma)

  implicit def toRichMultiArray2Int(ma: MultiArray2[Int]): RichMultiArray2Int = new RichMultiArray2Int(ma)

  implicit def toRichMultiArray2Double(ma: MultiArray2[Double]): RichMultiArray2Double = new RichMultiArray2Double(ma)

  implicit def toRichMutableMap[K, V](m: mutable.Map[K, V]): RichMutableMap[K, V] = new RichMutableMap(m)

  implicit def toRichOption[T](o: Option[T]): RichOption[T] = new RichOption[T](o)

  implicit def toRichOrderedArray[T: Ordering](a: Array[T]): RichOrderedArray[T] = new RichOrderedArray(a)

  implicit def toRichOrderedSeq[T: Ordering](s: Seq[T]): RichOrderedSeq[T] = new RichOrderedSeq[T](s)

  implicit def toRichPairRDD[K, V](r: RDD[(K, V)])(implicit kct: ClassTag[K],
    vct: ClassTag[V]): RichPairRDD[K, V] = new RichPairRDD(r)

  implicit def toRichVariantPairRDD[V](r: RDD[(Variant, V)])(implicit vct: ClassTag[V]): RichVariantPairRDD[V] =
    RichVariantPairRDD(r)

  implicit def toRichPairTraversableOnce[K, V](t: TraversableOnce[(K, V)]): RichPairTraversableOnce[K, V] =
    new RichPairTraversableOnce[K, V](t)

  implicit def toRichRDD[T](r: RDD[T])(implicit tct: ClassTag[T]): RichRDD[T] = new RichRDD(r)

  implicit def toRichRDDByteArray(r: RDD[Array[Byte]]): RichRDDByteArray = new RichRDDByteArray(r)

  implicit def toRichRow(r: Row): RichRow = new RichRow(r)

  implicit def toRichSC(sc: SparkContext): RichSparkContext = new RichSparkContext(sc)

  implicit def toRichSQLContext(sqlContext: SQLContext): RichSQLContext = new RichSQLContext(sqlContext)

  implicit def toRichSortedPairIterator[K, V](it: Iterator[(K, V)]): RichPairIterator[K, V] = new RichPairIterator(it)

  implicit def toRichString(str: String): RichString = new RichString(str)

  implicit def toRichStringBuilder(sb: mutable.StringBuilder): RichStringBuilder = new RichStringBuilder(sb)

  implicit def toRichStorageLevel(sl: StorageLevel): RichStorageLevel = new RichStorageLevel(sl)

  implicit def toTruncatable(s: String): Truncatable = s.truncatable()

  implicit def toTruncatable[T](it: Iterable[T]): Truncatable = it.truncatable()

  implicit def toTruncatable(arr: Array[_]): Truncatable = toTruncatable(arr: Iterable[_])

  implicit def toJSONWritable[T](x: T)(implicit jw: JSONWriter[T]): JSONWritable[T] = new JSONWritable(x, jw)

  implicit def toRichJValue(jv: JValue): RichJValue = new RichJValue(jv)
}

object NumericJavaLang {

  implicit object NumericJavaLangInt extends Numeric[java.lang.Integer] {

    def plus(x: java.lang.Integer, y: java.lang.Integer): java.lang.Integer = new java.lang.Integer(x.intValue() + y.intValue())

    def minus(x: java.lang.Integer, y: java.lang.Integer): java.lang.Integer = new java.lang.Integer(x.intValue() - y.intValue())

    def times(x: java.lang.Integer, y: java.lang.Integer): java.lang.Integer = new java.lang.Integer(x.intValue() * y.intValue())

    def negate(x: java.lang.Integer): java.lang.Integer = new java.lang.Integer(-x.intValue())

    def fromInt(x: Int): java.lang.Integer = new java.lang.Integer(x)

    def toInt(x: java.lang.Integer): Int = x.intValue()

    def toLong(x: java.lang.Integer): Long = x.longValue()

    def toFloat(x: java.lang.Integer): Float = x.floatValue()

    def toDouble(x: java.lang.Integer): Double = x.doubleValue()

    def compare(x: java.lang.Integer, y: java.lang.Integer): Int = x.intValue() - y.intValue()
  }

  implicit object NumericJavaLangLong extends Numeric[java.lang.Long] {

    def plus(x: java.lang.Long, y: java.lang.Long): java.lang.Long = new java.lang.Long(x.longValue() + y.longValue())

    def minus(x: java.lang.Long, y: java.lang.Long): java.lang.Long = new java.lang.Long(x.longValue() - y.longValue())

    def times(x: java.lang.Long, y: java.lang.Long): java.lang.Long = new java.lang.Long(x.longValue() * y.longValue())

    def negate(x: java.lang.Long): java.lang.Long = new java.lang.Long(-x.longValue())

    def fromInt(x: Int): java.lang.Long = new java.lang.Long(x)

    def toInt(x: java.lang.Long): Int = x.intValue()

    def toLong(x: java.lang.Long): Long = x.longValue()

    def toFloat(x: java.lang.Long): Float = x.floatValue()

    def toDouble(x: java.lang.Long): Double = x.doubleValue()

    def compare(x: java.lang.Long, y: java.lang.Long): Int = {
      val diff = x.longValue() - y.longValue()
      if (diff > 0) 1 else if (diff < 0) -1 else 0
    }  }

  implicit object NumericJavaLangFloat extends Numeric[java.lang.Float] {

    def plus(x: java.lang.Float, y: java.lang.Float): java.lang.Float = new java.lang.Float(x.floatValue() + y.floatValue())

    def minus(x: java.lang.Float, y: java.lang.Float): java.lang.Float = new java.lang.Float(x.floatValue() - y.floatValue())

    def times(x: java.lang.Float, y: java.lang.Float): java.lang.Float = new java.lang.Float(x.floatValue() * y.floatValue())

    def negate(x: java.lang.Float): java.lang.Float = new java.lang.Float(-x.floatValue())

    def fromInt(x: Int): java.lang.Float = new java.lang.Float(x)

    def toInt(x: java.lang.Float): Int = x.intValue()

    def toLong(x: java.lang.Float): Long = x.longValue()

    def toFloat(x: java.lang.Float): Float = x.floatValue()

    def toDouble(x: java.lang.Float): Double = x.doubleValue()

    def compare(x: java.lang.Float, y: java.lang.Float): Int = {
      val diff = x.floatValue() - y.floatValue()
      if (diff > 0) 1 else if (diff < 0) -1 else 0
    }
  }

  implicit object NumericJavaLangDouble extends Numeric[java.lang.Double] {

    def plus(x: java.lang.Double, y: java.lang.Double): java.lang.Double = new java.lang.Double(x.doubleValue() + y.doubleValue())

    def minus(x: java.lang.Double, y: java.lang.Double): java.lang.Double = new java.lang.Double(x.doubleValue() - y.doubleValue())

    def times(x: java.lang.Double, y: java.lang.Double): java.lang.Double = new java.lang.Double(x.doubleValue() * y.doubleValue())

    def negate(x: java.lang.Double): java.lang.Double = new java.lang.Double(-x.doubleValue())

    def fromInt(x: Int): java.lang.Double = new java.lang.Double(x)

    def toInt(x: java.lang.Double): Int = x.intValue()

    def toLong(x: java.lang.Double): Long = x.longValue()

    def toFloat(x: java.lang.Double): Float = x.floatValue()

    def toDouble(x: java.lang.Double): Double = x.doubleValue()

    def compare(x: java.lang.Double, y: java.lang.Double): Int = {
      val diff = x.doubleValue() - y.doubleValue()
      if (diff > 0) 1 else if (diff < 0) -1 else 0
    }
  }
}