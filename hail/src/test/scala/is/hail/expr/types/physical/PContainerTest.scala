package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Region, SafeIndexedSeq, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitMethodBuilder}
import is.hail.utils._
import org.codehaus.janino.util.Benchmark
import org.testng.annotations.Test

class PContainerTest extends HailSuite {
  def timeNano(fun: => Any): Long = {
    val start = System.nanoTime()
    fun
    System.nanoTime() - start
  }

  def nullInByte(nElements: Int, missingElement: Int) = {
    IndexedSeq.tabulate(nElements)(i => {
      if(i == missingElement - 1) {
        null
      } else {
        i + 1L
      }
    })
  }

  def testConvert(sourceType: PArray, destType: PArray, data: IndexedSeq[Any], expectFalse: Boolean) {
    val srcRegion = Region()
    val src = ScalaToRegionValue(srcRegion, sourceType, data)

    log.debug(s"Testing $data")

    val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
    val codeRegion = fb.getArg[Region](1).load()
    val value = fb.getArg[Long](2)

    fb.emit(destType.checkedConvertFrom(fb.apply_method, codeRegion, value, sourceType, "ShouldHaveNoNull"))

    val f = fb.result()()
    val destRegion = Region()
    if(expectFalse) {
      val thrown = intercept[Exception](f(destRegion,src))
      assert(thrown.getMessage == "ShouldHaveNoNull")
    } else {
      f(destRegion,src)
    }
  }

  @Test def checkedConvertFromTest() {
    val sourceType = PArray(PInt64(false))
    val destType = PArray(PInt64(true))

    // 1 byte
    testConvert(sourceType, destType, nullInByte(1, 0), false)
    testConvert(sourceType, destType, nullInByte(1, 1), true)
    testConvert(sourceType, destType, nullInByte(5, 5), true)

    // 1 full byte
    testConvert(sourceType, destType, nullInByte(8, 0), false)
    testConvert(sourceType, destType, nullInByte(8, 1), true)
    testConvert(sourceType, destType, nullInByte(8, 5), true)
    testConvert(sourceType, destType, nullInByte(8, 8), true)

    // 1 byte + remainder
    testConvert(sourceType, destType, nullInByte(11, 0), false)
    testConvert(sourceType, destType, nullInByte(13, 13), true)
    testConvert(sourceType, destType, nullInByte(13, 9), true)
    testConvert(sourceType, destType, nullInByte(13, 8), true)

    // 1 Long
    testConvert(sourceType, destType, nullInByte(64, 0), false)
    testConvert(sourceType, destType, nullInByte(64, 1), true)
    testConvert(sourceType, destType, nullInByte(64, 64), true)

    // 1 Long + remainder
    testConvert(sourceType, destType, nullInByte(67, 0), false)
    testConvert(sourceType, destType, nullInByte(67, 67), true)
    testConvert(sourceType, destType, nullInByte(67, 65), true)
    testConvert(sourceType, destType, nullInByte(67, 64), true)

    // 1 Long + 1 byte + remainder
    testConvert(sourceType, destType, nullInByte(79, 8), true)
  }

  @Test def checkedConvertFromBench() {
    val sourceType = PArray(PInt64(false))
    val destType = PArray(PInt64(true))

    def linearConvertFrom(mb: EmitMethodBuilder, r: Code[Region], value: Code[Long], otherPT: PType, msg: String): Code[Long] = {
      val otherPTA = otherPT.asInstanceOf[PArray]
      assert(otherPTA.elementType.isPrimitive && destType.isOfType(otherPTA))
      val oldOffset = value
      val len = otherPTA.loadLength(oldOffset)
      if (otherPTA.elementType.required == destType.elementType.required) {
        value
      } else {
        val newOffset = mb.newField[Long]
        Code(
          newOffset := destType.allocate(r, len),
          destType.stagedInitialize(newOffset, len),
          if (otherPTA.elementType.required) {
            // convert from required to non-required
            Code._empty
          } else {
            //  convert from non-required to required
            val i = mb.newField[Int]
            Code(
              i := 0,
              Code.whileLoop(i < len,
                otherPTA.isElementMissing(oldOffset, i).orEmpty(Code._fatal(s"${msg}: convertFrom $otherPT failed: element missing.")),
                i := i + 1
              )
            )
          },
          Region.copyFrom(otherPTA.elementOffset(oldOffset, len, 0), destType.elementOffset(newOffset, len, 0), len.toL * destType.elementByteSize),
          value
        )
      }
    }

    def timeOne(sourceType: PArray, destType: PArray, data: IndexedSeq[Any], fun: (EmitMethodBuilder, Code[Region], Code[Long], PContainer, String) => Code[Long]): Long = {
      val srcRegion = Region()
      val src = ScalaToRegionValue(srcRegion, sourceType, data)

      log.debug(s"Testing $data")

      val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
      val codeRegion = fb.getArg[Region](1).load()
      val value = fb.getArg[Long](2)

      fb.emit(fun(fb.apply_method, codeRegion, value, sourceType, "ShouldHaveNoNull"))
      val f = fb.result()()
      val linearTime = timeNano(try{ f(Region(),src) } catch { case _: Exception => {} })

      linearTime
    }

    def stats[T: Numeric](data: IndexedSeq[T]): (Long, Long, Long) = {
      val median = {
        val mod = data.length % 2
        val middleIdx: Int = data.length / 2 - 1

        if(mod == 1) {
          data(middleIdx).asInstanceOf[Long]
        } else {
          (data(middleIdx).asInstanceOf[Long] + data(middleIdx + 1).asInstanceOf[Long]) / 2
        }
      }

      (data(0).asInstanceOf[Long], data(data.length - 1).asInstanceOf[Long], median)
    }

    def compare(sourceType: PArray, destType: PArray, data: IndexedSeq[Any], iterations: Int): Float = {
      (0 to 50).map(_ => timeOne(sourceType: PArray, destType: PArray, data, linearConvertFrom))
      val linearTime = (0 to iterations).map(_ => timeOne(sourceType: PArray, destType: PArray, data, linearConvertFrom)).sorted
      val (lMin1, lMax1, lMed1) = stats(linearTime)

      Thread.sleep(500) // cpu cooldown

      (0 to 50).map(_ => timeOne(sourceType: PArray, destType: PArray, data, destType.checkedConvertFrom))
      val chunkedTime1 = (0 to iterations).map(_ => timeOne(sourceType: PArray, destType: PArray, data, destType.checkedConvertFrom)).sorted
      val (cMin1, cMax1, cMed1) = stats(chunkedTime1)

      Thread.sleep(500)

      (0 to 50).map(_ => timeOne(sourceType: PArray, destType: PArray, data, destType.checkedConvertFrom))
      val chunkedTime2 = (0 to iterations).map(_ => timeOne(sourceType: PArray, destType: PArray, data, destType.checkedConvertFrom)).sorted
      val (cMin2, cMax2, cMed2) = stats(chunkedTime2)

      Thread.sleep(500)

      (0 to 50).map(_ => timeOne(sourceType: PArray, destType: PArray, data, linearConvertFrom))
      val linearTime2 = (0 to iterations).map(_ => timeOne(sourceType: PArray, destType: PArray, data, linearConvertFrom)).sorted
      val (lMin2, lMax2, lMed2) = stats(linearTime2)

      val cMin = (cMin1 + cMin2) / 2
      val cMax = (cMax1 + cMax2) / 2
      val cMed = (cMed1 + cMed2) / 2

      val lMin = (lMin1 + lMin2) / 2
      val lMax = (lMax1 + lMax2) / 2
      val lMed = (lMed1 + lMed2) / 2

      println(s"cMin: $cMin")
      println(s"cMax: $cMax")
      println(s"cMed: $cMed")

      println(s"lMin: $lMin")
      println(s"lMax: $lMax")
      println(s"lMed: $lMed")

      lMed.toFloat / cMed.toFloat
    }

    var speedup = compare(sourceType, destType, nullInByte(200000, 200000), 100)
    println(s"Median speedup for last element Missing: $speedup")
    assert(speedup > 1)

    speedup = compare(sourceType, destType, nullInByte(200000, 3000), 100)
    println(s"Median speedup for element 3000 missing: $speedup")
    assert(speedup > 0)

    speedup = compare(sourceType, destType, nullInByte(200000, 1), 100)
    println(s"Median speedup for first element missing: $speedup")
    assert(speedup > 0)

    speedup = compare(sourceType, destType, nullInByte(200000, 0), 100)
    println(s"Median speedup for no element missing: $speedup")
    assert(speedup > 1)

    speedup = compare(sourceType, destType, nullInByte(200000, 80000), 100)
    println(s"Median speedup for 80,000 element missing: $speedup")
    assert(speedup > 1)

    speedup = compare(sourceType, destType, nullInByte(200000, 100000), 100)
    println(s"Median speedup for middle element missing: $speedup")
    assert(speedup > 1)

    speedup = compare(sourceType, destType, nullInByte(200000, 120000), 100)
    println(s"Median speedup for 120,000 missing: $speedup")
    assert(speedup > 1)

    speedup = compare(sourceType, destType, nullInByte(200003, 200000), 100)
    println(s"Median speedup for odd bits: $speedup")
    assert(speedup > 1)
  }
}