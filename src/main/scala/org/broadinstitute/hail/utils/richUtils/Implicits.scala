package org.broadinstitute.hail.utils.richUtils

import breeze.linalg.DenseMatrix
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}

import scala.collection.{TraversableOnce, mutable}
import scala.language.implicitConversions
import scala.reflect.ClassTag

trait Implicits {
  implicit def toRichAny(a: Any): RichAny = new RichAny(a)

  implicit def toRichArray[T](a: Array[T]): RichArray[T] = new RichArray(a)

  implicit def toRichArrayBuilderOfByte(t: mutable.ArrayBuilder[Byte]): RichArrayBuilderOfByte =
    new RichArrayBuilderOfByte(t)

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

  implicit def toRichMutableMap[K, V](m: mutable.Map[K, V]): RichMutableMap[K, V] = new RichMutableMap(m)

  implicit def toRichOption[T](o: Option[T]): RichOption[T] = new RichOption[T](o)

  implicit def toRichOrderedArray[T: Ordering](a: Array[T]): RichOrderedArray[T] = new RichOrderedArray(a)

  implicit def toRichOrderedSeq[T: Ordering](s: Seq[T]): RichOrderedSeq[T] = new RichOrderedSeq[T](s)

  implicit def toRichPairRDD[K, V](r: RDD[(K, V)])(implicit kct: ClassTag[K],
    vct: ClassTag[V]): RichPairRDD[K, V] = new RichPairRDD(r)

  implicit def toRichPairTraversableOnce[K, V](t: TraversableOnce[(K, V)]): RichPairTraversableOnce[K, V] =
    new RichPairTraversableOnce[K, V](t)

  implicit def toRichRDD[T](r: RDD[T])(implicit tct: ClassTag[T]): RichRDD[T] = new RichRDD(r)

  implicit def toRichRDDByteArray(r: RDD[Array[Byte]]): RichRDDByteArray = new RichRDDByteArray(r)

  implicit def toRichRow(r: Row): RichRow = new RichRow(r)

  implicit def toRichSC(sc: SparkContext): RichSparkContext = new RichSparkContext(sc)

  implicit def toRichSQLContext(sqlContext: SQLContext): RichSQLContext = new RichSQLContext(sqlContext)

  implicit def toRichSortedPairIterator[K, V](it: Iterator[(K, V)]): RichPairIterator[K, V] = new RichPairIterator(it)

  implicit def toRichStringBuilder(sb: mutable.StringBuilder): RichStringBuilder = new RichStringBuilder(sb)

}
