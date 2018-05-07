package is.hail.sparkextras

import is.hail._
import org.testng.annotations.Test

class ContextRDDSuite extends SparkSuite {
  val intplus = (x: Int, y: Int) => x + y
  val cintplus = (_: TrivialContext, x: Int, y: Int) => x + y
  def intplusarraylength[T] = (x: Int, y: Array[T]) => x + y.length
  def cintplusarraylength[T] =
    (_: TrivialContext, x: Int, y: Array[T]) => x + y.length

  private[this] def intDatasets(partitions: Int) = Seq(
    "1,2,3,4" -> Seq(1,2,3,4),
    "0 until 256" -> (0 until 256)
  ).map { case (name, localData) =>
    (name, ContextRDD.parallelize[TrivialContext](sc, localData, partitions))
  }

  @Test
  def aggregateSumVariousPartitionings() {
    for {
      partitions <- Seq(1, 4, 5)
      (name, data) <- intDatasets(partitions)
    } {
      assert(
        data.aggregate[Int](0, (_, x, y) => x + y, _ + _) == data.run.collect().sum,
        s"name, partitions: $partitions")
    }
  }

  @Test
  def treeAggregateVariousPartitionings() {
    for {
      partitions <- Seq(1, 15, 16, 64)
      (name, data) <- intDatasets(partitions)
      depth <- Seq(2, 3, 5)
    } {
      assert(
        data.treeAggregate[Int](0, cintplus, intplus, depth = depth) == data.run.collect().sum,
        s"$name, partitions: $partitions, depth: $depth")
    }
  }

  private[this] def seqDatasets(partitions: Int) = Seq(
    "1, 2, 3, 4" -> Seq(1, 2, 3, 4),
    "0 until 256" -> (0 until 256)
  ).map { case (name, lengths) =>
    (name, lengths.map(Array.fill(_)(0)))
  }.map { case (name, localData) =>
    (name, ContextRDD.parallelize[TrivialContext](sc, localData, partitions))
  }

  @Test
  def aggregateDistinctTandUTypes() {
    for {
      partitions <- Seq(1, 4, 5)
      (name, data) <- seqDatasets(partitions)
    } {
      assert(
        data.aggregate[Int](
          0,
          cintplusarraylength[Int],
          _ + _)
          ==
          data.run.collect().map(_.length).sum,
        s"$name, partitions: $partitions")
    }
  }

  @Test
  def treeAggregateDistinctTandUTypes() {
    for {
      partitions <- Seq(1, 15, 16, 64)
      (name, data) <- seqDatasets(partitions)
      depth <- Seq(2, 3, 5)
    } {
      assert(
        data.treeAggregate[Int](
          0,
          cintplusarraylength[Int],
          intplus,
          depth = 3)
          ==
          data.run.collect().map(_.length).sum,
        s"$name, partitions: $partitions")
    }
  }

  @Test
  def treeReduceOnEmptyIsError() {
    val data = ContextRDD.empty[TrivialContext, Int](sc)
    intercept[RuntimeException] {
      data.treeReduce(_ + _)
    }
  }

  def treeReduceOnSingletonIsSingleton() {
    val data = ContextRDD.parallelize[TrivialContext](sc, Seq(1))
    assert(data.treeReduce(_ + _) == 1)
  }

  @Test
  def treeReducePlusIsSum() {
    for {
      partitions <- Seq(1, 2, 3, 4, 5)
      (numbers, depth) <- Seq(
        (0 until 4) -> 2,
        (0 until 32) -> 2,
        (0 until 512) -> 5
      )
    } {
      val data = ContextRDD.parallelize[TrivialContext](sc, numbers, partitions)
      assert(
        data.treeReduce(_ + _, depth = depth) == numbers.sum,
        s"$numbers, $depth")
    }
  }
}
