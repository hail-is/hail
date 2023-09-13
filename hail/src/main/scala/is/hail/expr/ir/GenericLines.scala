package is.hail.expr.ir

import is.hail.backend.spark.SparkBackend
import is.hail.io.compress.BGzipInputStream
import is.hail.io.fs.{FS, FileListEntry, Positioned, PositionedInputStream, BGZipCompressionCodec}
import is.hail.io.tabix.{TabixReader, TabixLineIterator}
import is.hail.types.virtual.{TBoolean, TInt32, TInt64, TString, TStruct, Type}
import is.hail.utils._
import is.hail.variant.Locus

import org.apache.commons.io.input.{CountingInputStream, ProxyInputStream}
import org.apache.hadoop.io.compress.SplittableCompressionCodec
import org.apache.spark.{Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.annotation.meta.param

trait CloseableIterator[T] extends Iterator[T] with AutoCloseable

object CloseableIterator {
  def empty[T]: CloseableIterator[T] = new CloseableIterator[T] {
    def close(): Unit = ()
    def hasNext: Boolean = false
    def next: T = throw new NoSuchElementException
  }
}

object GenericLines {
  def read(fs: FS, contexts: IndexedSeq[Any], gzAsBGZ: Boolean, filePerPartition: Boolean): GenericLines = {

    val body: (FS, Any) => CloseableIterator[GenericLine] = { (fs: FS, context: Any) =>
      val contextRow = context.asInstanceOf[Row]
      val index = contextRow.getAs[Int](0)
      val fileNum = contextRow.getAs[Int](1)
      val file = contextRow.getAs[String](2)
      val start = contextRow.getAs[Long](3)
      val end = contextRow.getAs[Long](4)
      val split = contextRow.getAs[Boolean](5)

      new CloseableIterator[GenericLine] {
        private var splitCompressed = false
        private val is: PositionedInputStream = {
          val rawIS = fs.openNoCompression(file)
          val codec = fs.getCodecFromPath(file, gzAsBGZ)
          if (codec == null) {
            assert(split || filePerPartition)
            rawIS.seek(start)
            rawIS
          } else if (codec == BGZipCompressionCodec) {
            assert(split || filePerPartition)
            splitCompressed = true
            val bgzIS = new BGzipInputStream(rawIS, start, end, SplittableCompressionCodec.READ_MODE.BYBLOCK)
            new ProxyInputStream(bgzIS) with Positioned {
              def getPosition: Long = bgzIS.getVirtualOffset
            }
          } else {
            assert(!split || filePerPartition)
            new CountingInputStream(codec.makeInputStream(rawIS)) with Positioned {
              def getPosition: Long = getByteCount
            }
          }
        }

        private var eof = false
        private var closed = false

        private var buf = new Array[Byte](64 * 1024)
        private var bufOffset = 0L
        private var bufMark = 0
        private var bufPos = 0

        private var realEnd =
          if (splitCompressed)
            -1L  // end really means first block >= end
          else
            end

        private def loadBuffer(): Unit = {
          // compressed blocks can be empty
          while (bufPos == bufMark && !eof) {
            // load new block
            bufOffset = is.getPosition
            val nRead = is.read(buf)
            if (nRead == -1) {
              eof = true
              assert(!closed)
              close()
            } else {
              bufPos = 0
              bufMark = nRead
              assert(!splitCompressed || virtualOffsetBlockOffset(bufOffset) == 0)

              if (realEnd == -1 && bufOffset >= end)
                realEnd = bufOffset
            }
          }
        }

        loadBuffer()

        private var lineData = new Array[Byte](1024)

        private var line = new GenericLine(file, fileNum)
        line.data = lineData

        private def readLine(): Unit = {
          assert(line != null)

          if (eof) {
            line = null
            return
          }
          assert(bufPos < bufMark)

          val offset = bufOffset + bufPos
          if (split && realEnd != -1L && offset > realEnd) {
            line = null
            return
          }

          var sawcr = false
          var linePos = 0

          while (true) {
            if (eof) {
              assert(linePos > 0)
              line.setLine(offset, linePos)
              return
            }

            assert(bufPos < bufMark)

            val begin = bufPos
            var eol = false

            if (sawcr) {
              val c = buf(bufPos)
              if (c == '\n')
                bufPos += 1
              eol = true
            } else {
              // look for end of line in buf
              while (bufPos < bufMark && {
                val c = buf(bufPos)
                c != '\n' && c != '\r'
              })
                bufPos += 1

              if (bufPos < bufMark) {
                val c = buf(bufPos)
                if (c == '\n') {
                  bufPos += 1
                  eol = true
                } else {
                  assert(c == '\r')

                  bufPos += 1
                  if (bufPos < bufMark) {
                    val c2 = buf(bufPos)
                    if (c2 == '\n')
                      bufPos += 1
                    eol = true
                  } else
                    sawcr = true
                }
              }
            }

            // move scanned input from buf to lineData
            val n = bufPos - begin
            if (linePos + n > lineData.length) {
              val copySize = linePos.toLong + n

              // Maximum array size compatible with common JDK implementations
              // https://github.com/openjdk/jdk14u/blob/84917a040a81af2863fddc6eace3dda3e31bf4b5/src/java.base/share/classes/jdk/internal/util/ArraysSupport.java#L577
              val maxArraySize = Int.MaxValue - 8
              if (copySize > maxArraySize)
                fatal(s"GenericLines: line size reached: cannot read a line with more than 2^31-1 bytes")
              val newSize = Math.min(copySize * 2, maxArraySize).toInt
              if (newSize > (1 << 20)) {
                log.info(s"GenericLines: growing line buffer to $newSize")
              }

              val newLineData = new Array[Byte](newSize)
              System.arraycopy(lineData, 0, newLineData, 0, linePos)
              lineData = newLineData
              line.data = newLineData
            }
            System.arraycopy(buf, begin, lineData, linePos, n)
            linePos += n

            if (bufPos == bufMark)
              loadBuffer()

            if (eol) {
              assert(linePos > 0)
              line.setLine(offset, linePos)
              return
            }
          }
        }

        readLine()
        // the first line begins at most at start
        // belongs to previous partition
        if (index > 0 && line != null)
          readLine()

        private var consumed = false

        def hasNext: Boolean = {
          if (consumed) {
            readLine()
            consumed = false
          }
          line != null
        }

        def next(): GenericLine = {
          if (consumed)
            readLine()
          assert(line != null)
          assert(line.lineLength > 0)
          consumed = true

          line
        }

        def close(): Unit = {
          if (!closed) {
            is.close()
            closed = true
          }
        }
      }
    }

    val contextType = TStruct(
      "partitionIndex" -> TInt32,
      "fileNum" -> TInt32,
      "file" -> TString,
      "start" -> TInt64,
      "end" -> TInt64,
      "split" -> TBoolean)
    new GenericLines(
      contextType,
      contexts,
      body)
  }


  def read(
    fs: FS,
    fileListEntries0: IndexedSeq[FileListEntry],
    nPartitions: Option[Int],
    blockSizeInMB: Option[Int],
    minPartitions: Option[Int],
    gzAsBGZ: Boolean,
    allowSerialRead: Boolean,
    filePerPartition: Boolean = false
  ): GenericLines = {
    val fileListEntries = fileListEntries0.zipWithIndex.filter(_._1.getLen > 0)
    val totalSize = fileListEntries.map(_._1.getLen).sum

    var totalPartitions = nPartitions match {
      case Some(nPartitions) => nPartitions
      case None =>
        val blockSizeInB = blockSizeInMB.getOrElse(128) * 1024 * 1024
        (totalSize.toDouble / blockSizeInB + 0.5).toInt
    }
    minPartitions match {
      case Some(minPartitions) =>
        if (totalPartitions < minPartitions)
          totalPartitions = minPartitions
      case None =>
    }

    val contexts = fileListEntries.flatMap { case (fileListEntry, fileNum) =>
      val size = fileListEntry.getLen
      val codec = fs.getCodecFromPath(fileListEntry.getPath, gzAsBGZ)

      val splittable = codec == null || codec == BGZipCompressionCodec
      if (splittable && !filePerPartition) {
        var fileNParts = ((totalPartitions.toDouble * size) / totalSize + 0.5).toInt
        if (fileNParts == 0)
          fileNParts = 1

        val parts = partition(size, fileNParts)
        val partScan = parts.scanLeft(0L)(_ + _)
        Iterator.range(0, fileNParts)
          .map { i =>
            val start = partScan(i)
            var end = partScan(i + 1)
            if (codec != null)
              end = makeVirtualOffset(end, 0)
            Row(i, fileNum, fileListEntry.getPath, start, end, true)
          }
      } else {
        if (!allowSerialRead && !filePerPartition)
          fatal(s"Cowardly refusing to read file serially: ${ fileListEntry.getPath }.")

        Iterator.single {
          Row(0, fileNum, fileListEntry.getPath, 0L, size, false)
        }
      }
    }

    GenericLines.read(fs, contexts, gzAsBGZ, filePerPartition)
  }

  def readTabix(fs: FS, path: String, contigMapping: Map[String, String], partitions: IndexedSeq[Interval]): GenericLines = {
    val reverseContigMapping: Map[String, String] =
      contigMapping.toArray
        .groupBy(_._2)
        .map { case (target, mappings) =>
          if (mappings.length > 1)
            fatal(s"contig_recoding may not map multiple contigs to the same target contig, " +
              s"due to ambiguity when querying the tabix index." +
            s"\n  Duplicate mappings: ${ mappings.map(_._1).mkString(",") } all map to ${ target }")
            (target, mappings.head._1)
        }.toMap
    val contexts = partitions.zipWithIndex.map { case (interval, i) =>
      val start = interval.start.asInstanceOf[Row].getAs[Locus](0)
      val end = interval.end.asInstanceOf[Row].getAs[Locus](0)
      val contig = reverseContigMapping.getOrElse(start.contig, start.contig)
      Row(i, path, contig, start.position, end.position)
    }
    val body: (FS, Any) => CloseableIterator[GenericLine] = { (fs: FS, context: Any) =>
      val contextRow = context.asInstanceOf[Row]
      val index = contextRow.getAs[Int](0)
      val file = contextRow.getAs[String](1)
      val chrom = contextRow.getAs[String](2)
      val start = contextRow.getAs[Int](3)
      val end = contextRow.getAs[Int](4)

      val reg = {
        val r = new TabixReader(file, fs)
        val tid = r.chr2tid(chrom)
        r.queryPairs(tid, start - 1, end)
      }
      if (reg.isEmpty) {
        CloseableIterator.empty
      } else {
        new CloseableIterator[GenericLine] {
          private[this] val lines = new TabixLineIterator(fs, file, reg)
          private[this] var l = lines.next()
          private[this] var curIdx: Long = lines.getCurIdx()
          private[this] val inner = new Iterator[GenericLine] {
            def hasNext: Boolean = l != null
            def next(): GenericLine = {
              assert(l != null)
              val n = l
              val idx = curIdx
              l = lines.next()
              curIdx = lines.getCurIdx()
              if (l == null)
                lines.close()
              val bytes = n.getBytes
              new GenericLine(file, 0, idx, bytes, bytes.length)
            }
          }.filter { gl =>
            val s = gl.toString
            val t1 = s.indexOf('\t')
            val t2 = s.indexOf('\t', t1 + 1)

            val chr = s.substring(0, t1)
            val pos = s.substring(t1 + 1, t2).toInt

            if (chr != chrom) {
              throw new RuntimeException(s"bad chromosome! ${chrom}, $s")
            }
            start <= pos && pos <= end
          }

          def hasNext: Boolean = inner.hasNext
          def next(): GenericLine = inner.next()
          def close(): Unit = lines.close()
        }
      }
    }
    val contextType = TStruct(
      "partitionIndex" -> TInt32,
      "file" -> TString,
      "chrom" -> TString,
      "start" -> TInt32,
      "end" -> TInt32)
    new GenericLines(contextType, contexts, body)
  }

  def collect(fs: FS, lines: GenericLines): IndexedSeq[String] = {
    lines.contexts.flatMap { context =>
      using(lines.body(fs, context)) { it =>
        it.map(_.toString).toArray
      }
    }
  }
}

class GenericLine(
  val file: String,
  val fileNum: Int,
  // possibly virtual
  private var _offset: Long,
  var data: Array[Byte],
  private var _lineLength: Int) {
  def this(file: String, fileNum: Int) = this(file, fileNum, 0, null, 0)

  private var _str: String = null

  def setLine(newOffset: Long, newLength: Int): Unit = {
    _offset = newOffset
    _lineLength = newLength
    _str = null
  }

  def offset: Long = _offset

  def lineLength: Int = _lineLength

  override def toString: String = {
    if (_str == null) {
      var n = lineLength
      assert(n > 0)
      // strip line delimiter to match behavior of Spark textFile
      if (data(n - 1) == '\n') {
        n -= 1
        if (n > 0 && data(n - 1) == '\r')
          n -= 1
      } else if (data(n - 1) == '\r')
        n -= 1
      _str = new String(data, 0, n)
    }
    _str
  }
}

class GenericLinesRDDPartition(val index: Int, val context: Any) extends Partition

class GenericLinesRDD(
  @(transient@param) contexts: IndexedSeq[Any],
  body: (Any) => CloseableIterator[GenericLine]
) extends RDD[GenericLine](SparkBackend.sparkContext("GenericLinesRDD"), Seq()) {

  protected def getPartitions: Array[Partition] =
    contexts.iterator.zipWithIndex.map { case (c, i) =>
      new GenericLinesRDDPartition(i, c)
    }.toArray

  def compute(split: Partition, context: TaskContext): Iterator[GenericLine] = {
    val it = body(split.asInstanceOf[GenericLinesRDDPartition].context)
    TaskContext.get.addTaskCompletionListener[Unit] { _ =>
      it.close()
    }
    it
  }
}

class GenericLines(
  val contextType: Type,
  val contexts: IndexedSeq[Any],
  val body: (FS, Any) => CloseableIterator[GenericLine]) {

  def nPartitions: Int = contexts.length

  def toRDD(fs: FS): RDD[GenericLine] = {
    val localBody = body
    new GenericLinesRDD(contexts, localBody(fs, _))
  }
}
