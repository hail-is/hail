package is.hail.annotations

import is.hail.SparkSuite
import is.hail.asm4s.{Code, _}
import is.hail.asm4s.Code._
import is.hail.expr._
import is.hail.utils._
import org.testng.annotations.Test

class StagedRegionValueSuite extends SparkSuite {

  val showRVInfo = true

  @Test
  def testString() {
    val rt = TString
    val input = "hello"
    val srvb = new StagedRegionValueBuilder[String](FunctionBuilder.functionBuilder[String, MemoryBuffer, Long], rt)

    srvb.emit(srvb.start())
    srvb.emit(srvb.addString(srvb.input))
    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "string")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.addString(input)
    rv.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "string")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)
  }

  @Test
  def testInt() {
    val rt = TInt32
    val input = 3
    val srvb = new StagedRegionValueBuilder[Int](FunctionBuilder.functionBuilder[Int, MemoryBuffer, Long], rt)

    srvb.emit(srvb.start())
    srvb.emit(srvb.addInt32(srvb.input))
    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "int")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.addInt(input)
    rv.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "int")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)
  }

  @Test
  def testArray() {
    val rt = TArray(TInt32)
    val input = 3
    val srvb = new StagedRegionValueBuilder[Int](FunctionBuilder.functionBuilder[Int, MemoryBuffer, Long], TArray(TInt32))

    srvb.emit(
      Array[Code[_]](
        srvb.start(1),
        srvb.addInt32(srvb.input),
        srvb.advance()
      )
    )
    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "array")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.startArray(1)
    rvb.addInt(input)
    rvb.endArray()
    rv.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)


  }

  @Test
  def testStruct() {
    val rt = TStruct("a" -> TString, "b" -> TInt32)
    val input = 3
    val srvb = new StagedRegionValueBuilder[Int](FunctionBuilder.functionBuilder[Int, MemoryBuffer, Long], rt)

    srvb.emit(srvb.start())
    srvb.emit(srvb.addString("hello"))
    srvb.emit(srvb.advance())
    srvb.emit(srvb.addInt32(srvb.input))
    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "struct")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.startStruct()
    rvb.addString("hello")
    rvb.addInt(input)
    rvb.endStruct()
    rv.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "struct")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)
  }

  @Test
  def testArrayOfStruct() {
    val rt = TArray(TStruct("a"->TInt32, "b"->TString))
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[String, MemoryBuffer, Long]
    val srvb = new StagedRegionValueBuilder[String](fb, rt)

    val struct = { ssb: StagedRegionValueBuilder[String] =>
      Code(
        ssb.start(),
        ssb.addInt32(srvb.idx + 1),
        ssb.advance(),
        ssb.addString(ssb.input)
      )
    }

    srvb.emit(
      Array[Code[_]](
        srvb.start(2),
        Code.whileLoop(srvb.idx < 2,
          Code(
            srvb.addStruct(rt.elementType.asInstanceOf[TStruct], struct),
            srvb.advance()
          )
        )
      )
    )

    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "array of struct")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.startArray(2)
    for (i <- 1 to 2) {
      rvb.startStruct()
      rvb.addInt(i)
      rvb.addString(input)
      rvb.endStruct()
    }
    rvb.endArray()
    rv2.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "array of struct")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)

  }

  @Test
  def testStructWithArray() {
    val rt = TStruct("a"->TString, "b"->TArray(TInt32))
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[String, MemoryBuffer, Long]
    val srvb = new StagedRegionValueBuilder[String](fb, rt)

    val array = { sab: StagedRegionValueBuilder[String] =>
      Code(
        sab.start(2),
        Code.whileLoop(sab.idx < 2,
          Code(
            sab.addInt32(sab.idx + 1),
            sab.advance()
          )
        )
      )
    }

    srvb.emit(
      Array[Code[_]](
        srvb.start(),
        srvb.addString(srvb.input),
        srvb.advance(),
        srvb.addArray(TArray(TInt32), array)
      )
    )

    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "struct with array")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.startStruct()
    rvb.addString(input)
    rvb.startArray(2)
    for (i <- 1 to 2) {
      rvb.addInt(i)
    }
    rvb.endArray()
    rvb.endStruct()

    rv2.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "struct with array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)
  }

  @Test
  def testMissingArray() {
    val rt = TArray(TInt32)
    val input = 3
    val srvb = new StagedRegionValueBuilder[Int](FunctionBuilder.functionBuilder[Int, MemoryBuffer, Long], rt)

    srvb.emit(
      Array[Code[_]](
        srvb.start(2),
        srvb.addInt32(srvb.input),
        srvb.advance(),
        srvb.setMissing(),
        srvb.advance()
      )
    )
    srvb.build()

    val region = MemoryBuffer()
    val rv = RegionValue(region)
    rv.setOffset(srvb.transform()(input, region))

    if (showRVInfo) {
      printRegionValue(region, "missing array")
      println(rv.pretty(rt))
    }

    val region2 = MemoryBuffer()
    val rv2 = RegionValue(region2)
    val rvb = new RegionValueBuilder(region2)

    rvb.start(rt)
    rvb.startArray(2)
    rvb.addInt(input)
    rvb.setMissing()
    rvb.endArray()
    rv.setOffset(rvb.end())

    if (showRVInfo) {
      printRegionValue(region2, "missing array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(rv.offset == rv2.offset)


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
