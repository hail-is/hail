package is.hail.sparkextras

import is.hail.annotations.{JoinedRegionValue, RegionValue}

import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

package object implicits {

  implicit def toRichRowIterator(it: Iterator[Row]): RichRowIterator = new RichRowIterator(it)

  implicit def toRichPairRDD[K, V](r: RDD[(K, V)]): RichPairRDD[K, V] = new RichPairRDD(r)

  implicit def toRichRDD[T](r: RDD[T]): RichRDD[T] = new RichRDD(r)

  implicit def toRichContextRDDRegionValue(r: ContextRDD[RegionValue]): RichContextRDDRegionValue =
    new RichContextRDDRegionValue(r)

  implicit def toRichContextRDD[A](r: ContextRDD[A]): RichContextRDD[A] =
    new RichContextRDD(r)

  implicit def toRichContextRDDPair[K, V](r: ContextRDD[(K, V)]): RichContextRDDPair[K, V] =
    new RichContextRDDPair(r)

  implicit def toRichContextRDDLong(r: ContextRDD[Long]): RichContextRDDLong =
    new RichContextRDDLong(r)

  implicit def toRichContextRDDRow(r: ContextRDD[Row]): RichContextRDDRow =
    new RichContextRDDRow(r)

  implicit def toRichRow(r: Row): RichRow = new RichRow(r)

  implicit def toRichSC(sc: SparkContext): RichSparkContext = new RichSparkContext(sc)

  implicit def toRichTaskContext(ctx: TaskContext): RichTaskContext =
    new RichTaskContext(ctx)

  implicit def toRichJoinedRegionValue(jrv: JoinedRegionValue): RichJoinedRegionValue =
    new RichJoinedRegionValue(jrv)
}
