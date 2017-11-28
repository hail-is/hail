package org.apache.spark.rdd

import org.apache.spark.util.Utils
import scala.reflect.ClassTag

object AggregateWithContext {
  implicit class RichRDD[T](val rdd: RDD[T]) extends AnyVal {
    def aggregateWithContext[U: ClassTag, V](context: => V)(zeroValue: U)
      (seqOp: (V, U, T) => U, combOp: (U, U) => U): U = rdd.withScope {
      val sc = rdd.sparkContext
      // Clone the zero value since we will also be serializing it as part of tasks
      var jobResult = Utils.clone(zeroValue, sc.env.serializer.newInstance())
      val cleanSeqOp = sc.clean(seqOp)
      val cleanCombOp = sc.clean(combOp)
      val aggregatePartition = { (it: Iterator[T]) =>
        val localContext = context
        it.aggregate(zeroValue)(cleanSeqOp(localContext, _, _), cleanCombOp)
      }
      val cleanAggregatePartition = sc.clean(aggregatePartition)
      val mergeResult = (index: Int, taskResult: U) => jobResult = combOp(jobResult, taskResult)
      sc.runJob(rdd, cleanAggregatePartition, mergeResult)
      jobResult
    }
  }
}
