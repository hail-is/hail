package is.hail.utils

import is.hail.SparkSuite
import is.hail.check.Arbitrary._
import is.hail.check.{Gen, Prop, Properties}
import is.hail.sparkextras._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s.jackson.JsonMethods
import org.testng.annotations.Test

import scala.math.Ordering.Implicits._

case class PartitionSummary(partitionIndex: Int,
  sorted: Boolean,
  correctPartitioning: Boolean,
  maybeBounds: Option[(Variant, Variant)])

class OrderedRDDSuite extends SparkSuite {

  object Spec extends Properties("OrderedRDDFunctions") {
    val v = for (pos <- Gen.choose(1, 1000);
      alt <- Gen.oneOfGen(Gen.const("T"), Gen.const("C"), Gen.const("G"), Gen.const("TTT"), Gen.const("ATA")))
      yield Variant("16", pos, "A", alt)

    val g = for (uniqueVariants <- Gen.buildableOf[Set, Variant](v).map(set => set.toIndexedSeq);
      toZip <- Gen.buildableOfN[IndexedSeq, String](uniqueVariants.size, arbitrary[String]);
      nPar <- Gen.choose(1, 10)) yield (nPar, uniqueVariants.zip(toZip))

    val random = for ((n, v) <- g;
      shuffled <- Gen.shuffle(v)) yield sc.parallelize(shuffled, n)

    val locusSorted = for ((n, v) <- g;
      locusSorted <- Gen.const(v.sortBy(_._1.locus))) yield sc.parallelize(locusSorted, n)

    val scrambledInPartition = locusSorted.map(_.mapPartitions { it => it.toArray.reverseIterator })

    val sorted = for ((n, v) <- g;
      sorted <- Gen.const(v.sortBy(_._1))) yield sc.parallelize(sorted, n)

    def check(rdd: OrderedRDD[Locus, Variant, String], original: RDD[(Variant, String)]): Boolean = {
      val p = rdd.orderedPartitioner

      val partitionSummaries = rdd.mapPartitionsWithIndex { case (partitionIndex, iter) =>
        val a = iter.toArray
        val keys = a.map(_._1)
        val sorted = keys.isSorted
        val correctPartitioning = keys.forall(k => p.getPartition(k) == partitionIndex)
        Iterator(PartitionSummary(partitionIndex, sorted, correctPartitioning,
          if (a.nonEmpty) Some((a.head._1, a.last._1)) else None))
      }.collect().sortBy(_.partitionIndex)

      val sameElems = rdd.collect.toSet == original.collect.toSet

      val validPartitions = partitionSummaries.flatMap(_.maybeBounds.map(_._1)).headOption match {
        case Some(first) =>
          val sortedWithin = partitionSummaries.forall(_.sorted)
          val partitionedCorrectly = partitionSummaries.forall(_.correctPartitioning)
          val sortedBetween = partitionSummaries.flatMap(_.maybeBounds)
            .tail
            .foldLeft((true, first)) { case ((b, last), (start, end)) =>
              (b && start > last, end)
            }._1
          val p = sortedWithin && partitionedCorrectly && sortedBetween
          if (!p) {
            println(s"sortedWithin: $sortedWithin")
            println(s"sortedBetween: $sortedBetween")
            println(s"partitionedCorrectly: $partitionedCorrectly")
          }
          p
        case None => true
      }

      sameElems && validPartitions
    }

    def checkLeftJoin(nPar1: Int, is1: IndexedSeq[(Variant, String)],
      nPar2: Int, is2: IndexedSeq[(Variant, String)]): Boolean = {

      val leftDistinct = is1.toMap.toIndexedSeq
      val m2 = is2.groupBy(_._1).mapValues(_.map(_._2).toSet)

      val rdd1 = sc.parallelize(leftDistinct, nPar1).cache()
      val rdd2 = sc.parallelize(is2, nPar2).cache()

      val leftJoin = rdd1.toOrderedRDD
        .orderedLeftJoin(rdd2.toOrderedRDD)
        .collect()
        .toIndexedSeq

      val check1 = leftDistinct.toSet == leftJoin.map { case (k, (v1, _)) => (k, v1) }.toSet
      val check2 = leftJoin.forall { case (k, (_, v2)) =>
        val eq = v2.toSet == m2.getOrElse(k, Set.empty[String])
        if (!eq)
          println(
            s"""key=$k
               |  v1 = ${ v2.toSet }
               |  v2 = ${ m2.getOrElse(k, Set.empty[String]) }""".stripMargin)
        eq
      }

      val p = check1 && check2
      if (!p)
        println(
          s"""check1 : $check1
             |check2 : $check2""".stripMargin)
      p
    }

    def checkLeftJoinDistinct(nPar1: Int, is1: IndexedSeq[(Variant, String)],
      nPar2: Int, is2: IndexedSeq[(Variant, String)]): Boolean = {
      val m2 = is2.toMap

      val rdd1 = sc.parallelize(is1, nPar1).cache()
      val rdd2 = sc.parallelize(is2, nPar2).cache()

      val leftJoin: IndexedSeq[(Variant, (String, Option[String]))] = rdd1.toOrderedRDD
        .orderedLeftJoinDistinct(rdd2.toOrderedRDD)
        .collect()
        .toIndexedSeq

      val check1 = is1.toSet == leftJoin.map { case (k, (v1, _)) => (k, v1) }.toSet
      val check2 = leftJoin.forall { case (k, (_, v2)) => v2 == m2.get(k) }
      val check3 = rdd1.leftOuterJoinDistinct(rdd2).collect().toMap == leftJoin.toMap

      val outerJoin: IndexedSeq[(Variant, (Option[String], Option[String]))] = rdd1.toOrderedRDD
        .orderedOuterJoinDistinct(rdd2.toOrderedRDD)
        .collect()
        .toIndexedSeq

      val map1 = is1.toMap
      val map2 = is2.toMap

      val check4 = is1.toSet == outerJoin.flatMap { case (k, (v1, _)) => v1.map(v => (k, v)) }.toSet
      val check5 = is2.toSet == outerJoin.flatMap { case (k, (_, v2)) => v2.map(v => (k, v)) }.toSet
      val check6 = outerJoin.toSet == (map1.keySet ++ map2.keySet).map(k => (k, (map1.get(k), map2.get(k))))

      val p = check1 && check2 && check3 && check4 && check5 && check6
      if (!p)
        println(
          s"""check1 : $check1
             |check2 : $check2
             |check3 : $check3
             |check4 : $check4
             |check5 : $check5
             |check6 : $check6""".stripMargin)
      p
    }

    property("randomlyOrdered") = Prop.forAll(random) { rdd =>
      val (_, ordered) = OrderedRDD.coerce(rdd)
      check(ordered, rdd)
    }

    property("scrambledInPartition") = Prop.forAll(scrambledInPartition) { rdd =>
      val (status, ordered) = OrderedRDD.coerce(rdd)
      check(ordered, rdd) && status <= OrderedRDD.SHUFFLE
    }

    property("locusSorted") = Prop.forAll(locusSorted) { rdd =>
      val (status, ordered) = OrderedRDD.coerce(rdd)
      check(ordered, rdd) && status <= OrderedRDD.LOCAL_SORT
    }

    property("fullySorted") = Prop.forAll(sorted) { rdd =>
      val (status, ordered) = OrderedRDD.coerce(rdd)
      // could be local sort if variant is split across partitions
      check(ordered, rdd) && status <= OrderedRDD.LOCAL_SORT
    }

    property("join1") = Prop.forAll(g, g) { case ((nPar1, is1), (nPar2, is2)) =>
      checkLeftJoinDistinct(nPar1, is1, nPar2, is2)
    }

    property("leftProductJoin") = {
      val localV = for {
        pos <- Gen.choose(1, 50)
        alt <- Gen.oneOf("T", "C")
      } yield Variant("1", pos, "A", alt)

      val localG = for (variants <- Gen.buildableOf[Array, Variant](localV).map(x => x: IndexedSeq[Variant]);
        toZip <- Gen.buildableOfN[Array, String](variants.size, arbitrary[String]);
        nPar <- Gen.choose(1, 10)) yield (nPar, variants.zip(toZip))

      Prop.forAll(localG, localG) { case ((n1, left), (n2, right)) => checkLeftJoin(n1, left, n2, right) }
    }

    // check different levels of partition skipping and sparsity
    val v2 = for (pos <- Gen.oneOfGen(Gen.choose(1, 100), Gen.choose(400, 500), Gen.choose(900, 1000));
      alt <- Gen.oneOfGen(Gen.const("T"), Gen.const("C"), Gen.const("G"), Gen.const("TTT"), Gen.const("ATA")))
      yield Variant("16", pos, "A", alt)

    val g2 = for (uniqueVariants <- Gen.buildableOf[Set, Variant](v2).map(set => set.toIndexedSeq);
      toZip <- Gen.buildableOfN[IndexedSeq, String](uniqueVariants.size, arbitrary[String]);
      nPar <- Gen.choose(1, 10)) yield (nPar, uniqueVariants.zip(toZip))

    property("join2") = Prop.forAll(g, g2) { case ((nPar1, is1), (nPar2, is2)) =>
      checkLeftJoinDistinct(nPar1, is1, nPar2, is2)
    }

    property("join3") = Prop.forAll(g, g2) { case ((_, is1), (nPar2, is2)) =>
      checkLeftJoinDistinct(1, is1, nPar2, is2)
    }

    property("join4") = Prop.forAll(g2, g) { case ((nPar1, is1), (nPar2, is2)) =>
      checkLeftJoinDistinct(nPar1, is1, nPar2, is2)
    }

    property("join4") = Prop.forAll(g2, g) { case ((_, is1), (nPar2, is2)) =>
      checkLeftJoinDistinct(1, is1, nPar2, is2)
    }

    val scrambled = for (rdd <- Gen.oneOfGen(sorted, locusSorted);
      newPartitions <- Gen.shuffle(rdd.partitions.indices)
    ) yield (rdd, rdd.reorderPartitions(newPartitions.toArray))

    property("partitionReordering") = Prop.forAll(scrambled) { case (original, scrambled) =>

      val (status, result) = OrderedRDD.coerce(original)
      val (status2, result2) = OrderedRDD.coerce(scrambled)

      status == status2 && result.fullOuterJoin(result2)
        .forall { case (v, (s1, s2)) => s1 == s2 }
    }
  }

  @Test def test() {
    Spec.check()
  }

  @Test def testEmptyPartitions() {
    val emptyPartitions = sc.parallelize(Array[(Variant, String)](), 10)
    val (status, rdd) = OrderedRDD.coerce(emptyPartitions)
    assert(status == OrderedRDD.AS_IS)
  }
}

