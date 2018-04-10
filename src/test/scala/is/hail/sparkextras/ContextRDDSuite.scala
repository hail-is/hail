package is.hail.sparkextras

import is.hail._
import org.testng.annotations.Test

class ContextRDDSuite extends SparkSuite {
  val intplus = (x: Int, y: Int) => x + y
  def intplusarraylength[T] = (x: Int, y: Array[T]) => x + y.length

  @Test
  def aggregateSumVariousPartitionings() {
    for {
      (name, data) <- Seq(
        "one partition" ->
          ContextRDD.parallelize[TrivialContext](sc, Seq(1,2,3,4), 1),
        "more partitions than records" ->
          ContextRDD.parallelize[TrivialContext](sc, Seq(1,2,3,4), 5),
        "lots of records, indivisble number of partitions" ->
          ContextRDD.parallelize[TrivialContext](sc, 0 until 256, 5),
        "lots of records, divisble number of partitions" ->
          ContextRDD.parallelize[TrivialContext](sc, 0 until 256, 16)
      )
    } {
      assert(
        data.aggregate[Int](0, _ + _, _ + _) == data.run.collect().sum,
        name)
    }
  }

  @Test
  def treeAggregateVariousPartitionings() {
    for {
      (name, data, depth) <- Seq(
        ("one partition",
          ContextRDD.parallelize[TrivialContext](sc, Seq(1,2,3,4), 1),
          2
        ),
        ("more partitions than records",
          ContextRDD.parallelize[TrivialContext](sc, Seq(1,2,3,4), 5),
          2
        ),
        ("lots of records, indivisble number of partitions",
          ContextRDD.parallelize[TrivialContext](sc, 0 until 256, 15),
          2
        ),
        ("lots of records, divisble number of partitions",
          ContextRDD.parallelize[TrivialContext](sc, 0 until 256, 16),
          2
        ),
        ("lots of records, divisble number of partitions, depth 3",
          ContextRDD.parallelize[TrivialContext](sc, 0 until 256, 16),
          3
        ),
        ("lots of records, indivisble number of partitions, depth 3",
          ContextRDD.parallelize[TrivialContext](sc, 0 until 256, 15),
          2
        )
      )
    } {
      assert(
        data.treeAggregate[Int](0, intplus, intplus, depth = depth) == data.run.collect().sum,
        name)
    }
  }

  @Test
  def aggregateDistinctTandUTypes() {
    for {
      (name, data) <- Seq(
        "indivisble number of partitions" ->
          ContextRDD.parallelize[TrivialContext](
            sc,
            (0 until 256).map(Array.fill(_)(0)),
            5),
        "divisble number of partitions" ->
          ContextRDD.parallelize[TrivialContext](
            sc,
            (0 until 256).map(Array.fill(_)(0)),
            5)
      )
    } {
      assert(
        data.aggregate[Int](0, _ + _.length, _ + _)
          ==
          data.run.collect().map(_.length).sum,
        name)
    }
  }

  @Test
  def treeAggregateDistinctTandUTypes() {
    for {
      (name, data) <- Seq(
        "indivisble number of partitions" ->
          ContextRDD.parallelize[TrivialContext](
            sc,
            (0 until 256).map(Array.fill(_)(0)),
            5),
        "divisble number of partitions" ->
          ContextRDD.parallelize[TrivialContext](
            sc,
            (0 until 256).map(Array.fill(_)(0)),
            5)
      )
    } {
      assert(
        data.treeAggregate[Int](
          0,
          intplusarraylength[Int],
          intplus,
          depth = 3)
          ==
          data.run.collect().map(_.length).sum,
        name)
    }
  }
}
