package is.hail.sparkextras

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.rvd.{PartitionBoundOrdering, RVD, RVDContext, RVDPartitioner, RVDType}
import is.hail.utils._
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec

object OrderedDependency {
  def generate[T](oldPartitioner: RVDPartitioner, newIntervals: IndexedSeq[Interval], rdd: RDD[T]): OrderedDependency[T] = {
    new OrderedDependency(
      newIntervals.map(oldPartitioner.queryInterval).toArray,
      rdd)
  }
}

class OrderedDependency[T](
  depArray: Array[Range],
  rdd: RDD[T]
) extends NarrowDependency[T](rdd) {

  override def getParents(partitionId: Int): Seq[Int] = depArray(partitionId)
}

object RepartitionedOrderedRDD2 {
  def apply(sm: HailStateManager, prev: RVD, newRangeBounds: IndexedSeq[Interval]): ContextRDD[Long] =
    ContextRDD(new RepartitionedOrderedRDD2(sm, prev, newRangeBounds))
}

/**
  * Repartition 'prev' to comply with 'newRangeBounds', using narrow dependencies.
  * Assumes new key type is a prefix of old key type, so no reordering is
  * needed.
  */
class RepartitionedOrderedRDD2 private (sm: HailStateManager, @transient val prev: RVD, @transient val newRangeBounds: IndexedSeq[Interval])
  extends RDD[ContextRDD.ElementType[Long]](prev.crdd.sparkContext, Nil) { // Nil since we implement getDependencies

  val prevCRDD: ContextRDD[Long] = prev.crdd
  val typ: RVDType = prev.typ
  val kOrd: ExtendedOrdering = PartitionBoundOrdering(sm, typ.kType.virtualType)


  def getPartitions: Array[Partition] = {
    require(newRangeBounds.forall{i => typ.kType.virtualType.relaxedTypeCheck(i.start) && typ.kType.virtualType.relaxedTypeCheck(i.end)})
    Array.tabulate[Partition](newRangeBounds.length) { i =>
      RepartitionedOrderedRDD2Partition(
        i,
        dependency.getParents(i).toArray.map(prevCRDD.partitions),
        newRangeBounds(i))
    }
  }

  override def compute(partition: Partition, context: TaskContext): Iterator[RVDContext => Iterator[Long]] = {
    val ordPartition = partition.asInstanceOf[RepartitionedOrderedRDD2Partition]
    val pord = kOrd.intervalEndpointOrdering
    val range = ordPartition.range

    Iterator.single { (outerCtx: RVDContext) =>
      new Iterator[Long] {
        private[this] val innerCtx = outerCtx.freshContext()
        private[this] val outerRegion = outerCtx.region
        private[this] val innerRegion = innerCtx.region
        private[this] val parentIterator = ordPartition.parents.iterator.flatMap(p => prevCRDD.iterator(p, context).flatMap(_.apply(innerCtx)))
        private[this] var pulled: Boolean = false
        private[this] var current: Long =  _
        private[this] val ur = new UnsafeRow(typ.rowType)
        private[this] val key = new SelectFieldsRow(ur, typ.kFieldIdx)

        // drop left elements at iterator allocation to avoid extra control flow in hasNext()
        dropLeft()

        @tailrec private[this] def dropLeft(): Unit = {
          if (parentIterator.hasNext) {
            pull()
            if (pord.lt(key, range.left)) {
              innerRegion.clear()
              dropLeft()
            } else if (pord.gt(key, range.right))
              // End the iterator if first remaining value is greater than range.right
              end()
          } else
            // End the iterator if we exhausted parent iterators before finding an element greater than range.left
            end()
        }

        private[this] def pull(): Unit = {
          current = parentIterator.next()
          ur.set(innerRegion, current)
          pulled = true
        }

        private[this] def end(): Unit = {
          pulled = false
          innerRegion.clear()
        }

        def hasNext: Boolean = {
          if (pulled)
            return true

          if (!parentIterator.hasNext)
            return false

          pull()

          if (pord.gt(key, range.right)) {
            end()
            return false
          }

          true
        }

        def next(): Long = {
          // hasNext() must be called before next() to fill `current`
          if (!hasNext)
            throw new NoSuchElementException
          pulled = false
          innerRegion.move(outerRegion)
          current
        }

      }
    }
  }

  val dependency: OrderedDependency[_] = OrderedDependency.generate(
    prev.partitioner,
    newRangeBounds,
    prevCRDD.rdd)

  override def getDependencies: Seq[Dependency[_]] = FastSeq(dependency)
}

case class RepartitionedOrderedRDD2Partition(
    index: Int,
    parents: Array[Partition],
    range: Interval
) extends Partition
