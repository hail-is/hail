package is.hail.expr.ir

import is.hail.utils._
import is.hail.expr.types.virtual.{TBoolean, TInt32, TInt64, TString, TStruct, Type}
import is.hail.io.compress.{BGzipCodec, BGzipInputStream}
import is.hail.io.fs.{FS, Positioned, PositionedInputStream, SeekableInputStream}
import org.apache.commons.io.input.{CountingInputStream, ProxyInputStream}
import org.apache.hadoop.io.compress.SplittableCompressionCodec
import org.apache.spark.sql.Row

abstract class CloseableIterator[T] extends Iterator[T] with AutoCloseable

object GenericLines {
  def read(fs: FS, contexts: IndexedSeq[Any]): GenericLines = {

    val fsBc = fs.broadcast
    val body: (Any) => CloseableIterator[GenericLine] = { (context: Any) =>
      val contextRow = context.asInstanceOf[Row]
      val index = contextRow.getAs[Int](0)
      val file = contextRow.getAs[String](1)
      val start = contextRow.getAs[Long](2)
      val end = contextRow.getAs[Long](3)
      val split = contextRow.getAs[Boolean](4)

      new CloseableIterator[GenericLine] {
        private var splitCompressed = false
        private val is: PositionedInputStream = {
          val fs = fsBc.value
          val rawIS = fs.openNoCompression(file)
          val codec = fs.getCodec(file)
          if (codec == null) {
            rawIS.seek(start)
            rawIS
          } else if (codec.isInstanceOf[BGzipCodec]) {
            splitCompressed = true
            val bgzIS = new BGzipInputStream(rawIS, start, end, SplittableCompressionCodec.READ_MODE.BYBLOCK)
            new ProxyInputStream(bgzIS) with Positioned {
              def getPosition: Long = bgzIS.getVirtualOffset
            }
          } else {
            assert(!split)
            new CountingInputStream(codec.createInputStream(rawIS)) with Positioned {
              def getPosition: Long = getByteCount
            }
          }
        }

        private var eof = false

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
            if (nRead == -1)
              eof = true
            else {
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

        private var line = new GenericLine(file)
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

          line.offset = offset

          var sawcr = false
          var linePos = 0

          while (true) {
            if (eof) {
              assert(linePos > 0)
              line.lineLength = linePos
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
              val newLineData = new Array[Byte]((linePos + n) * 2)
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
              line.lineLength = linePos
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
          is.close()
        }
      }
    }

    val contextType = TStruct(
      "index" -> TInt32,
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
    files: IndexedSeq[String],
    blockSizeInMB: Option[Int],
    nPartitions: Option[Int],
    allowSerialRead: Boolean
  ): GenericLines = {
    val statuses = fs.globAllStatuses(files)
      .filter(_.getLen > 0)
    val totalSize = statuses.map(_.getLen).sum

    val contexts = statuses.flatMap { status =>
      val size = status.getLen
      val codec = fs.getCodec(status.getPath)

      val splittable = codec == null || codec.isInstanceOf[BGzipCodec]
      if (splittable) {
        var fileNParts = nPartitions match {
          case Some(nPartitions) =>
            ((nPartitions.toDouble * size) / totalSize + 0.5).toInt
          case None =>
            val blockSizeInB = blockSizeInMB.getOrElse(128) * 1024 * 1024
            (size.toDouble / blockSizeInB + 0.5).toInt
        }
        if (fileNParts == 0)
          fileNParts = 1

        val parts = partition(size, fileNParts)
        val partScan = parts.scanLeft(0L)(_ + _)
        Iterator.range(0, fileNParts)
          .map { i =>
            var start = partScan(i)
            var end = partScan(i + 1)
            if (codec != null)
              end = makeVirtualOffset(end, 0)
            Row(i, status.getPath, start, end, true)
          }
      } else {
        if (!allowSerialRead)
          fatal(s"Cowardly refusing to read file serially: ${ status.getPath }.")

        Iterator.single {
          Row(0, status.getPath, 0L, size, false)
        }
      }
    }

    GenericLines.read(fs, contexts)
  }

  def collect(lines: GenericLines): IndexedSeq[String] = {
    lines.contexts.flatMap { context =>
      using(lines.body(context)) { it =>
        it.map { line =>
          var n = line.lineLength
          assert(n > 0)
          val lineData = line.data
          // strip line delimiter to match behavior of Spark textFile
          if (lineData(n - 1) == '\n') {
            n -= 1
            if (n > 0 && lineData(n - 1) == '\r')
              n -= 1
          } else if (lineData(n - 1) == '\r')
            n -= 1
          new String(lineData, 0, n)
        }.toArray
      }
    }
  }
}

class GenericLine(
  val file: String,
  // possibly virtual
  var offset: Long,
  var data: Array[Byte],
  var lineLength: Int) {
  def this(file: String) = this(file, 0, null, 0)
}

class GenericLines(
  val contextType: Type,
  val contexts: IndexedSeq[Any],
  val body: (Any) => CloseableIterator[GenericLine])
