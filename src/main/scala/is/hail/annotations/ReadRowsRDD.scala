package is.hail.annotations

import java.io.DataInputStream

import is.hail.expr.TStruct
import is.hail.utils.{SerializableHadoopConfiguration, _}
import net.jpountz.lz4.{LZ4BlockInputStream, LZ4Factory}
import org.apache.commons.lang3.StringUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, SparkContext, TaskContext}

class RichRDDRow(val rdd: RDD[Row]) extends AnyVal {
  def writeRows(path: String, t: TStruct) {
    val sc = rdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/rowstore")

    val tBc = sc.broadcast(t)
    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitions = rdd.partitions.length
    val d = digitsNeeded(nPartitions)

    val rowCount = rdd.mapPartitionsWithIndex { case (i, it) =>
      val region = MemoryBuffer(8 * 1024)
      val rvb = new RegionValueBuilder(region)
      val buffer = new Array[Byte](8 * 1024)
      var rowCount = 0L

      val t = tBc.value
      val f = t.fundamentalType

      val is = i.toString
      assert(is.length <= d)
      val pis = StringUtils.leftPad(is, d, "0")

      sHadoopConfBc.value.value.writeLZ4DataFile(path + "/rowstore/part-" + pis,
        64 * 1024,
        LZ4Factory.fastestInstance().highCompressor()) { out =>
        it.foreach { r =>
          region.clear()
          rvb.start(f)
          rvb.addRow(t, r)
          val offset = rvb.end()
          assert(offset == 0)

          val rowSize = region.offset
          out.writeInt(rowSize)

          var totalWritten = 0
          while (totalWritten < rowSize) {
            val toWrite = buffer.length.min(rowSize - totalWritten)
            region.loadBytes(totalWritten, toWrite, buffer)
            out.write(buffer, 0, toWrite)
            totalWritten += toWrite
          }

          rowCount += 1
        }

        out.writeInt(-1)
      }

      Iterator(rowCount)
    }
      .fold(0L)(_ + _)

    info(s"wrote $rowCount records")
  }
}

case class ReadRowsRDDPartition(index: Int) extends Partition

class ReadRowsRDD(sc: SparkContext,
  path: String, t: TStruct, nPartitions: Int) extends RDD[Row](sc, Nil) {
  val ttBc = BroadcastTypeTree(sc, t)

  override def getPartitions: Array[Partition] =
    Array.tabulate(nPartitions)(i => ReadRowsRDDPartition(i))

  private val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  override def compute(split: Partition, context: TaskContext): Iterator[UnsafeRow] = {
    val d = digitsNeeded(nPartitions)
    val localPath = path

    new Iterator[UnsafeRow] {
      private val in = {
        val is = split.index.toString
        assert(is.length <= d)
        val pis = StringUtils.leftPad(is, d, "0")
        new DataInputStream(
          new LZ4BlockInputStream(sHadoopConfBc.value.value.unsafeReader(localPath + "/rowstore/part-" + pis),
            LZ4Factory.fastestInstance().fastDecompressor()))
      }

      private var rowSize = in.readInt()

      private val buffer = new Array[Byte](8 * 1024)
      private val region = MemoryBuffer(rowSize.max(8 * 1024))

      def hasNext: Boolean = rowSize != -1

      def next(): UnsafeRow = {
        if (!hasNext)
          throw new NoSuchElementException("next on empty iterator")

        region.clear()
        region.ensure(rowSize)

        var totalRead = 0
        while (totalRead < rowSize) {
          val read = in.read(buffer, 0, (rowSize - totalRead).min(buffer.length))
          assert(read > 0)

          region.appendBytes(buffer, read)

          totalRead += read
        }
        assert(region.offset == rowSize)

        rowSize = in.readInt()
        if (rowSize == -1)
          in.close()

        new UnsafeRow(ttBc, region.copy(), 0)
      }
    }
  }
}
