package is.hail.annotations

import is.hail.SparkSuite
import is.hail.asm4s.{Code, _}
import is.hail.asm4s.Code._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._
import org.testng.annotations.Test

class StagedParserSuite extends SparkSuite {

  val showRVInfo = true

  def throwMsg[T](msg: Code[String]): Code[T] = _throw[Throwable, T](Code.newInstance[Throwable, String](msg))

  def storeAsByteArray(s: Code[String], line: LocalRef[Array[Byte]]): Code[Unit] = {
    line.store(s.invoke[Array[Byte]]("getBytes"))
  }

  def getNextString(s: Code[Array[Byte]], sep: Code[Byte], firstoffset: LocalRef[Int], offset: LocalRef[Int]): Code[String] = {
    Code(
      offset.store(firstoffset),
      whileLoop(offset < s.length() && s(offset).cne(sep),
        offset.store(offset + 1)
      ),
      Code.newInstance[String, Array[Byte], Int, Int](s, firstoffset, offset - firstoffset)
    )
  }

  def getNextInt(s: Code[Array[Byte]], sep: Code[Byte], firstoffset: LocalRef[Int], offset: LocalRef[Int], v: LocalRef[Int]): Code[Unit] = {
    Code(
      v.store(0),
      offset.store(firstoffset),
      s(offset).ceq(sep).mux(
        throwMsg[Unit]("two separator characters found next to each other"),
        _empty[Unit]
      ),
      (s(offset).ceq('-'.toInt) || s(offset).ceq('+'.toInt)).mux(
        offset.store(offset + 1),
        _empty[Unit]
      ),
      whileLoop(offset < s.length() && s(offset).cne(sep),
        Code(
          ((s(offset) <= '9'.toInt) && (s(offset) >= '0'.toInt)).mux(
            Code(
              v.store(v * 10),
              v.store(v + s(offset) - '0'.toInt)
            ),
            throwMsg[Unit]("found invalid character while parsing int")
          ),
          offset.store(offset + 1)
        )
      ),
      s(firstoffset).ceq('-'.toInt).mux(
        v.store(v.negate()),
        _empty
      )
    )
  }

  def isMissing(s: Code[Array[Byte]], isMissing: Code[Array[Byte]], sep: Code[Byte], firstoffset: LocalRef[Int], offset: LocalRef[Int]): Code[Boolean] = {
    Code(
      offset.store(firstoffset),
      whileLoop(offset - firstoffset < isMissing.length() && s(offset).ceq(isMissing(offset - firstoffset)),
        offset.store(offset + 1)
      ),
      ((offset - firstoffset).ceq(isMissing.length()) && s(offset).ceq(sep)).mux(
        true,
        false
      )
    )
  }

  def getStagedMatrixLines(input: Array[String], nSamples: Int, seperator: String, missingValue: String): Array[String] = {
    val rt = TStruct("a"->TString, "b"-> TArray(TInt32))
    val fb = FunctionBuilder.functionBuilder[String, MemoryBuffer, Long]
    val srvb = new StagedRegionValueBuilder[String](fb, rt)

    val line = fb.newLocal[Array[Byte]]
    val firstoff = fb.newLocal[Int]
    val offset = fb.newLocal[Int]
    val v = fb.newLocal[Int]

    val sep = fb.newLocal[Byte]
    val missingVal = fb.newLocal[Array[Byte]]

    srvb.emit(
      Array[Code[_]](
        srvb.start(),
        storeAsByteArray(srvb.input, line),
        storeAsByteArray(missingValue, missingVal),
        sep.store(seperator(0).toByte),
        firstoff.store(0),
        srvb.addString(getNextString(line, sep, firstoff, offset)),
        firstoff.store(offset+1),
        srvb.addArray(TArray(TInt32), { sab: StagedRegionValueBuilder[String] =>
          Code(
            sab.start(nSamples),
            whileLoop(sab.idx < nSamples,
              (firstoff >= line.length()).mux(
                throwMsg("found fewer samples than expected."),
                Code(
                  isMissing(line, missingVal, sep, firstoff, offset).mux(
                    sab.setMissing(),
                    Code(
                      getNextInt(line, sep, firstoff, offset, v),
                      sab.addInt32(v)
                    )
                  ),
                  firstoff.store(offset + 1)
                )
              )
            ),
            (firstoff < line.length()).mux(
              throwMsg("found more data than expected"),
              _empty
            )
          )
        })
      )
    )

    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)

    input.map(line => {
      region.clear()
      rv.setOffset(srvb.transform(line, region))
      if (showRVInfo) {
        printRegionValue(region, "TStruct")
        println(rv.pretty(rt))
      }
      rv.pretty(rt)
    })
  }

  @Test def testParseMatrix() {

    val input = Array("hello\t1\t2\t3","world\t4\tNA\t6")
    val nSamples = 3

    val sep = "\t"
    val missingValue = "NA"

    val staged = getStagedMatrixLines(input, nSamples, sep, missingValue)
    val unstaged = getUnstagedMatrixLines(input, nSamples, sep, missingValue)

    assert(staged(0) == unstaged(0))
    assert(staged(1) == unstaged(1))

  }


  def setInt32(string: String, off: Int, rvb: RegionValueBuilder, sep: String = "\t", missingValue: String = "NA"): Int = {
    var newoff = off
    var v = 0
    var isNegative = false
    if (string(off) == sep(0)) {
      return -1
    }
    if (string(off) == '-' || string(off) == '+') {
      isNegative = string(off) == '-'
      newoff += 1
    }
    while (newoff < string.length && string(newoff) >= '0' && string(newoff) <= '9') {
      v *= 10
      if (isNegative) {
        v -= string(newoff) - '0'
      } else {
        v += string(newoff) - '0'
      }
      newoff += 1
    }
    if (newoff == off) {
      while (newoff - off < missingValue.length && missingValue(newoff-off) == string(newoff)) {
        newoff += 1
      }
      if (newoff - off == missingValue.length && (string.length == newoff || string(newoff) == sep(0))) {
        rvb.setMissing()
        newoff
      } else { -1 }
    } else if (string.length == newoff || string(newoff) == sep(0)) {
      rvb.addInt(v)
      newoff
    } else { -1 }
  }

  def getUnstagedMatrixLines(input: Array[String], nSamples: Int, sep: String, missingValue: String): Array[String] = {
    val rt = TStruct("a"->TString, "b"->TArray(TInt32))

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    val rvb = new RegionValueBuilder(region)

    input.map { line =>

      if (line.nonEmpty) {
        val firstsep = line.indexOf(sep)

        region.clear()
        rvb.start(rt)
        rvb.startStruct()

        rvb.addString(line.substring(0, firstsep))

        rvb.startArray(nSamples)
        var off = firstsep + 1
        var ii = 0
        while (ii < nSamples) {
          if (off > line.length) {
            fatal(
              s"""Incorrect number of elements in line:
                 |    expected $nSamples elements in row but only $ii elements found""".stripMargin
            )
          }
          off = setInt32(line, off, rvb, sep, missingValue)
          ii += 1
          if (off == -1 || (off != line.length && line(off) != sep(0))) {
            fatal(
              s"""found a bad input in line $line: ii = $ii, off = $off, line(off) = ${ line(off) }""".stripMargin
            )
          }
          off += 1
        }
        if (off < line.length) {
          fatal(
            s"""Incorrect number of elements in line:
               |    expected $nSamples elements in row but more data found.""".stripMargin
          )
        }
        rvb.endArray()
        rvb.endStruct()
        rv.setOffset(rvb.end())
      }
      if (showRVInfo) {
        printRegionValue(region, "TStruct")
        println(rv.pretty(rt))
      }
      rv.pretty(rt)
    }
  }

  def printRegionValue(region:MemoryBuffer, string:String) {
    println(string)
    val size = region.size
    println("Region size: "+size.toString)
    val bytes = region.loadBytes(0,size.toInt)
    println("Array: ")
    var j = 0
    for (i <- bytes) {
      j += 1
      print(i)
      if (j % 30 == 0) {
        print('\n')
      } else {
        print('\t')
      }
    }
    print('\n')
  }

}