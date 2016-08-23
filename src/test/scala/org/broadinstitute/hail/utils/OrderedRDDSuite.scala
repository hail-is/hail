package org.broadinstitute.hail.utils

import org.apache.spark.rdd.{OrderedRDD, RDD}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.{OrderedPartitioner, Partition, SparkContext, TaskContext}
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Arbitrary._
import org.broadinstitute.hail.check.{Gen, Prop, Properties}
import org.broadinstitute.hail.variant.{Locus, Variant}
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
      shuffled <- Gen.shuffle(v)) yield (n, shuffled)

    val locusSorted = for ((n, v) <- g;
      locusSorted <- Gen.const(v.sortBy(_._1.locus))) yield (n, locusSorted)

    val sorted = for ((n, v) <- g;
      sorted <- Gen.const(v.sortBy(_._1))) yield (n, sorted)

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

      val join: IndexedSeq[(Variant, (String, Option[String]))] = rdd1.toOrderedRDD(_.locus)
        .orderedLeftJoinDistinct(rdd2.toOrderedRDD(_.locus))
        .collect()
        .toIndexedSeq

      val check1 = is1.toMap == join.map { case (k, (v1, _)) => (k, v1) }.toMap
      val check2 = join.forall { case (k, (_, v2)) => v2 == m2.get(k) }
      val check3 = rdd1.leftOuterJoinDistinct(rdd2).collect().toMap == join.toMap

      check1 && check2 && check3
    }

    property("randomlyOrdered") = Prop.forAll(random) { case (nPar, s) =>
      check(sc.parallelize(s, nPar).toOrderedRDD(_.locus))
    }

    property("locusSorted") = Prop.forAll(locusSorted) { case (nPar, s) =>
      check(sc.parallelize(s, nPar).toOrderedRDD(_.locus))
    }

    property("fullySorted") = Prop.forAll(sorted) { case (nPar, s) =>
      check(sc.parallelize(s, nPar).toOrderedRDD(_.locus))
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
      nPar <- Gen.choose(1, 25)) yield (nPar, uniqueVariants.zip(toZip))

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
      val rdd = sc.parallelize(is, nPar).toOrderedRDD(_.locus)
      val schema = StructType(Array(
        StructField("variant", Variant.schema, nullable = false),
        StructField("str", StringType, nullable = false)))
      hadoopDelete(tmpRdd, hadoopConf, recursive = true)
      val df = sqlContext.createDataFrame(rdd.map { case (v, s) => Row.fromSeq(Seq(v.toRow, s)) }, schema)
        .write.parquet(tmpRdd)

      writeObjectFile(tmpPartitioner, hadoopConf) { out =>
        rdd.partitioner.get.asInstanceOf[OrderedPartitioner[Variant, String]].write(out)
      }

      val status = hadoopFileStatus(tmpPartitioner, hadoopConf)

      val rddReadBack = sqlContext.readPartitioned.parquet(tmpRdd)
        .rdd
        .map(r => (Variant.fromRow(r.getAs[Row](0)), r.getAs[String](1)))

      val readBackPartitioner = readObjectFile(tmpPartitioner, hadoopConf) { in =>
        OrderedPartitioner.read[Locus, Variant](in, _.locus)
      }

      val orderedRddRB = new OrderedRDD[Locus, Variant, String](rddReadBack, readBackPartitioner)

      orderedRddRB.zipPartitions(rdd) { case (it1, it2) =>
        it1.zip(it2)
      }
        .collect()
        .forall { case (v1, v2) => v1 == v2 }
    }
  }

  @Test def test() {
    Spec.check()
  }
}

