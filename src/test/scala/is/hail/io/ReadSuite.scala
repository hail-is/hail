package is.hail.io

import is.hail.SparkSuite
import is.hail.utils._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.testng.annotations.Test

import scala.language.postfixOps

class ReadSuite extends SparkSuite {
  // verify loading parquet files >128MB with readParquetSorted aren't split
  @Test def test() {
    val file = tmpDir.createTempFile(extension = "parquet")

    // creates two ~256M parquet files
    val rdd = sc.parallelize(Seq(1, 2), 2)
      .mapPartitions { it =>
        it.flatMap { seed =>
          var x: Long = seed
          Iterator.fill(32 * 1024 * 1024) {
            x = 6364136223846793005L * x + 1
            x
          }
        }
      }
      .map { s =>
        Row(s)
      }
    assert(rdd.partitions.length == 2)

    val df = sqlContext.createDataFrame(rdd, StructType(Seq(
      StructField("s", LongType))))

    df.write.parquet(file)

    val rdd2 = sqlContext.readParquetSorted(file)
    assert(rdd2.partitions.length == 2)
  }
}
