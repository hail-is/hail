package org.broadinstitute.hail.utils

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.check.Arbitrary._
import org.broadinstitute.hail.check.{Gen, Prop, Properties}
import org.broadinstitute.hail.sparkextras.{OrderedPartitioner, _}
import org.broadinstitute.hail.variant._
import org.testng.annotations.Test

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

    val sorted = for ((n, v) <- g;
      sorted <- Gen.const(v.sortBy(_._1))) yield sc.parallelize(sorted, n)

    def check(rdd: OrderedRDD[Locus, Variant, String]): Boolean = {
      val p = rdd.orderedPartitioner
      val partitionSummaries = rdd.mapPartitionsWithIndex { case (partitionIndex, iter) =>
        val a = iter.toArray
        val keys = a.map(_._1)
        val sorted = keys.isSorted
        val correctPartitioning = keys.forall(k => p.getPartition(k) == partitionIndex)
        Iterator((partitionIndex, sorted, correctPartitioning, if (a.nonEmpty) Some((a.head._1, a.last._1)) else None))
      }.collect().sortBy(_._1)

      partitionSummaries.flatMap(_._4.map(_._1)).headOption match {
        case Some(first) =>
          val sortedWithin = partitionSummaries.forall(_._2)
          val partitionedCorrectly = partitionSummaries.forall(_._3)
          val sortedBetween = partitionSummaries.flatMap(_._4)
            .tail
            .foldLeft((true, first)) { case ((b, last), (start, end)) =>
              (b && start > last, end)
            }._1
          sortedWithin && partitionedCorrectly && sortedBetween
        case None => true
      }
    }


    def checkJoin(nPar1: Int, is1: IndexedSeq[(Variant, String)],
      nPar2: Int, is2: IndexedSeq[(Variant, String)]): Boolean = {
      val m2 = is2.toMap

      val rdd1 = sc.parallelize(is1, nPar1).cache()
      val rdd2 = sc.parallelize(is2, nPar2).cache()

      val join: IndexedSeq[(Variant, (String, Option[String]))] = rdd1.toOrderedRDD
        .orderedLeftJoinDistinct(rdd2.toOrderedRDD)
        .collect()
        .toIndexedSeq

      val check1 = is1.toMap == join.map { case (k, (v1, _)) => (k, v1) }.toMap
      val check2 = join.forall { case (k, (_, v2)) => v2 == m2.get(k) }
      val check3 = rdd1.leftOuterJoinDistinct(rdd2).collect().toMap == join.toMap

      check1 && check2 && check3
    }

    property("randomlyOrdered") = Prop.forAll(random) { rdd =>
      val (_, ordered) = OrderedRDD.coerce(rdd)
      check(ordered)
    }

    property("locusSorted") = Prop.forAll(locusSorted) { rdd =>
      val (status, ordered) = OrderedRDD.coerce(rdd)
      check(ordered) && status <= OrderedRDD.LOCAL_SORT
    }

    property("fullySorted") = Prop.forAll(sorted) { rdd =>
      val (status, ordered) = OrderedRDD.coerce(rdd)
      check(ordered) && status == OrderedRDD.AS_IS
    }

    property("join1") = Prop.forAll(g, g) { case ((nPar1, is1), (nPar2, is2)) =>
      checkJoin(nPar1, is1, nPar2, is2)
    }

    // check different levels of partition skipping and sparsity
    val v2 = for (pos <- Gen.oneOfGen(Gen.choose(1, 100), Gen.choose(400, 500), Gen.choose(900, 1000));
      alt <- Gen.oneOfGen(Gen.const("T"), Gen.const("C"), Gen.const("G"), Gen.const("TTT"), Gen.const("ATA")))
      yield Variant("16", pos, "A", alt)

    val g2 = for (uniqueVariants <- Gen.buildableOf[Set, Variant](v2).map(set => set.toIndexedSeq);
      toZip <- Gen.buildableOfN[IndexedSeq, String](uniqueVariants.size, arbitrary[String]);
      nPar <- Gen.choose(1, 10)) yield (nPar, uniqueVariants.zip(toZip))

    property("join2") = Prop.forAll(g, g2) { case ((nPar1, is1), (nPar2, is2)) =>
      checkJoin(nPar1, is1, nPar2, is2)
    }

    property("join3") = Prop.forAll(g, g2) { case ((_, is1), (nPar2, is2)) =>
      checkJoin(1, is1, nPar2, is2)
    }

    property("join4") = Prop.forAll(g2, g) { case ((nPar1, is1), (nPar2, is2)) =>
      checkJoin(nPar1, is1, nPar2, is2)
    }

    property("join4") = Prop.forAll(g2, g) { case ((_, is1), (nPar2, is2)) =>
      checkJoin(1, is1, nPar2, is2)
    }


    val tmpPartitioner = tmpDir.createTempFile("partitioner")
    val tmpRdd = tmpDir.createTempFile("rdd", ".parquet")

    property("writeRead") = Prop.forAll(g) { case (nPar, is) =>
      val rdd = sc.parallelize(is, nPar).toOrderedRDD
      val schema = StructType(Array(
        StructField("variant", Variant.schema, nullable = false),
        StructField("str", StringType, nullable = false)))
      hadoopConf.delete(tmpRdd, recursive = true)
      val df = sqlContext.createDataFrame(rdd.map { case (v, s) => Row.fromSeq(Seq(v.toRow, s)) }, schema)
        .write.parquet(tmpRdd)

      hadoopConf.writeObjectFile(tmpPartitioner) { out =>
        rdd.partitioner.get.asInstanceOf[OrderedPartitioner[Variant, String]].write(out)
      }

      val status = hadoopConf.fileStatus(tmpPartitioner)

      val rddReadBack = sqlContext.sortedParquetRead(tmpRdd)
        .map(r => (Variant.fromRow(r.getAs[Row](0)), r.getAs[String](1)))

      val readBackPartitioner = hadoopConf.readObjectFile(tmpPartitioner) { in =>
        OrderedPartitioner.read[Locus, Variant](in, rddReadBack.partitions.length)
      }

      val orderedRddRB = OrderedRDD[Locus, Variant, String](rddReadBack, readBackPartitioner)

      orderedRddRB.zipPartitions(rdd) { case (it1, it2) =>
        it1.zip(it2)
      }
        .collect()
        .forall { case (v1, v2) => v1 == v2 }
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

