package is.hail.annotations

import is.hail.HailSuite
import is.hail.asm4s.{Code, FunctionBuilder, _}
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir.{EmitFunctionBuilder, EmitRegion}
import org.testng.annotations.Test
import is.hail.asm4s.Code._
import is.hail.asm4s.FunctionBuilder.functionBuilder
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.Type
import is.hail.utils._
class StagedRegionValueSuite extends HailSuite {

  val showRVInfo = true

  @Test
  def testString() {
    val rt = PString()
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[Region, String, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString(fb.getArg[String](2)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "string")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    val bytes = input.getBytes()
    val boff = PBinary.allocate(region2, bytes.length)
    Region.storeInt(boff, bytes.length)
    Region.storeBytes(PBinary.bytesOffset(boff), bytes)
    rv2.setOffset(boff)

    if (showRVInfo) {
      printRegion(region2, "string")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(PString.loadString(rv.region, rv.offset) ==
      PString.loadString(rv2.region, rv2.offset))
  }

  @Test
  def testInt() {
    val rt = PInt32()
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addInt(fb.getArg[Int](2)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "int")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(region2.allocate(4, 4))
    Region.storeInt(rv2.offset, input)

    if (showRVInfo) {
      printRegion(region2, "int")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(Region.loadInt(rv.offset) == Region.loadInt(rv2.offset))
  }

  @Test
  def testArray() {
    val rt = PArray(PInt32())
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(1),
        srvb.addInt(fb.getArg[Int](2)),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "array")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(region2, rt, FastIndexedSeq(input)))

    if (showRVInfo) {
      printRegion(region2, "array")
      println(rv2.pretty(rt))
    }

    assert(rt.loadLength(rv.region, rv.offset) == 1)
    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(Region.loadInt(rt.loadElement(rv.region, rv.offset, 0)) ==
      Region.loadInt(rt.loadElement(rv2.region, rv2.offset, 0)))
  }

  @Test
  def testStruct() {
    val rt = PStruct("a" -> PString(), "b" -> PInt32())
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString("hello"),
        srvb.advance(),
        srvb.addInt(fb.getArg[Int](2)),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "struct")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(region2, rt, Annotation("hello", input)))

    if (showRVInfo) {
      printRegion(region2, "struct")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(PString.loadString(rv.region, rt.loadField(rv.region, rv.offset, 0)) ==
      PString.loadString(rv2.region, rt.loadField(rv2.region, rv2.offset, 0)))
    assert(Region.loadInt(rt.loadField(rv.region, rv.offset, 1)) ==
      Region.loadInt(rt.loadField(rv2.region, rv2.offset, 1)))
  }

  @Test
  def testArrayOfStruct() {
    val rt = PArray(PStruct("a" -> PInt32(), "b" -> PString()))
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[Region, String, Long]
    val srvb = new StagedRegionValueBuilder(fb, rt)

    val struct = { ssb: StagedRegionValueBuilder =>
      Code(
        ssb.start(),
        ssb.addInt(srvb.arrayIdx + 1),
        ssb.advance(),
        ssb.addString(fb.getArg[String](2))
      )
    }

    fb.emit(
      Code(
        srvb.start(2),
        Code.whileLoop(srvb.arrayIdx < 2,
          Code(
            srvb.addBaseStruct(rt.elementType.asInstanceOf[PStruct], struct),
            srvb.advance()
          )
        ),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "array of struct")
      println(rv.pretty(rt))
    }

    val region2 = Region()
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
      printRegion(region2, "array of struct")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)))
  }

  @Test
  def testMissingRandomAccessArray() {
    val rt = PArray(PStruct("a" -> PInt32(), "b" -> PString()))
    val intVal = 20
    val strVal = "a string with a partner of 20"
    val region = Region()
    val region2 = Region()
    val rvb = new RegionValueBuilder(region)
    val rvb2 = new RegionValueBuilder(region2)
    val rv = RegionValue(region)
    val rv2 = RegionValue(region2)
    rvb.start(rt)
    rvb.startMissingArray(4)
    rvb.setArrayIndex(2)
    rvb.setPresent()
    rvb.startStruct()
    rvb.addInt(intVal)
    rvb.addString(strVal)
    rvb.endStruct()
    rvb.endArrayUnchecked()
    rv.setOffset(rvb.end())

    rvb2.start(rt)
    rvb2.startArray(4)
    for (i <- 0 to 3) {
      if (i == 2) {
        rvb2.startStruct()
        rvb2.addInt(intVal)
        rvb2.addString(strVal)
        rvb2.endStruct()
      } else {
        rvb2.setMissing()
      }
    }
    rvb2.endArray()
    rv2.setOffset(rvb2.end())
    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)))
  }

  @Test
  def testSetFieldPresent() {
    val rt = PStruct("a" -> PInt32(), "b" -> PString(), "c" -> PFloat64())
    val intVal = 30
    val floatVal = 39.273d
    val r = Region()
    val r2 = Region()
    val rv = RegionValue(r)
    val rv2 = RegionValue(r2)
    val rvb = new RegionValueBuilder(r)
    val rvb2 = new RegionValueBuilder(r2)
    rvb.start(rt)
    rvb.startStruct()
    rvb.setMissing()
    rvb.setMissing()
    rvb.addDouble(floatVal)
    rvb.setFieldIndex(0)
    rvb.setPresent()
    rvb.addInt(intVal)
    rvb.setFieldIndex(3)
    rvb.endStruct()
    rv.setOffset(rvb.end())

    rvb2.start(rt)
    rvb2.startStruct()
    rvb2.addInt(intVal)
    rvb2.setMissing()
    rvb2.addDouble(floatVal)
    rvb2.endStruct()
    rv2.setOffset(rvb2.end())

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(Region.loadInt(rt.loadField(rv.region, rv.offset, 0)) ==
      Region.loadInt(rt.loadField(rv2.region, rv2.offset, 0)))
    assert(Region.loadDouble(rt.loadField(rv.region, rv.offset, 2)) ==
      Region.loadDouble(rt.loadField(rv2.region, rv2.offset, 2)))
  }

  @Test
  def testStructWithArray() {
    val rt = PStruct("a" -> PString(), "b" -> PArray(PInt32()))
    val input = "hello"
    val fb = FunctionBuilder.functionBuilder[Region, String, Long]
    val codeInput = fb.getArg[String](2)
    val srvb = new StagedRegionValueBuilder(fb, rt)

    val array = { sab: StagedRegionValueBuilder =>
      Code(
        sab.start(2),
        Code.whileLoop(sab.arrayIdx < 2,
          Code(
            sab.addInt(sab.arrayIdx + 1),
            sab.advance()
          )
        )
      )
    }

    fb.emit(
      Code(
        srvb.start(),
        srvb.addString(codeInput),
        srvb.advance(),
        srvb.addArray(rt.types(1).asInstanceOf[PArray], array),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "struct with array")
      println(rv.pretty(rt))
    }

    val region2 = Region()
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
      printRegion(region2, "struct with array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeRow(rt, rv.region, rv.offset) ==
      new UnsafeRow(rt, rv2.region, rv2.offset))
  }

  @Test
  def testMissingArray() {
    val rt = PArray(PInt32())
    val input = 3
    val fb = FunctionBuilder.functionBuilder[Region, Int, Long]
    val codeInput = fb.getArg[Int](2)
    val srvb = new StagedRegionValueBuilder(fb, rt)

    fb.emit(
      Code(
        srvb.start(2),
        srvb.addInt(codeInput),
        srvb.advance(),
        srvb.setMissing(),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val rv = RegionValue(region)
    rv.setOffset(fb.result()()(region, input))

    if (showRVInfo) {
      printRegion(region, "missing array")
      println(rv.pretty(rt))
    }

    val region2 = Region()
    val rv2 = RegionValue(region2)
    rv2.setOffset(ScalaToRegionValue(region2, rt, FastIndexedSeq(input, null)))

    if (showRVInfo) {
      printRegion(region2, "missing array")
      println(rv2.pretty(rt))
    }

    assert(rv.pretty(rt) == rv2.pretty(rt))
    assert(new UnsafeIndexedSeq(rt, rv.region, rv.offset).sameElements(
      new UnsafeIndexedSeq(rt, rv2.region, rv2.offset)))
  }

  def printRegion(region: Region, string: String) {
    println(region.prettyBits())
  }

  @Test
  def testAddPrimitive() {
    val t = PStruct("a" -> PInt32(), "b" -> PBoolean(), "c" -> PFloat64())
    val fb = FunctionBuilder.functionBuilder[Region, Int, Boolean, Double, Long]
    val srvb = new StagedRegionValueBuilder(fb, t)

    fb.emit(
      Code(
        srvb.start(),
        srvb.addIRIntermediate(PInt32())(fb.getArg[Int](2)),
        srvb.advance(),
        srvb.addIRIntermediate(PBoolean())(fb.getArg[Boolean](3)),
        srvb.advance(),
        srvb.addIRIntermediate(PFloat64())(fb.getArg[Double](4)),
        srvb.advance(),
        srvb.end()
      )
    )

    val region = Region()
    val f = fb.result()()
    def run(i: Int, b: Boolean, d: Double): (Int, Boolean, Double) = {
      val off = f(region, i, b, d)
      (Region.loadInt(t.loadField(region, off, 0)),
        Region.loadBoolean(t.loadField(region, off, 1)),
        Region.loadDouble(t.loadField(region, off, 2)))
    }

    assert(run(3, true, 42.0) == ((3, true, 42.0)))
    assert(run(42, false, -1.0) == ((42, false, -1.0)))
  }

  //GOAL: For an array either copy the array (deep copy), or just copy the pointer to the array if type is same
  //If types are the same, the value is just copied by pointer (addAddress)
  //If the types are different, the value needs to be copied not by pointer.
  // However in the array case, I think this just means we need a missing bit header and a pointer to the values
  // (the memory address of the values)
  // How to do this?
  // Well, if 1D array, we just grab the pointer, or grab the pointer and add a header
  // If we have an n-d array, we do this recursively
  // THen we also want a parameter of 'shallow', to allow people to perform an actual element-by-element copy
  @Test
  def testStructWithArray2() {
    val rt = PArray(PInt32(), false, true)
    val requiredRt = PArray(PInt32())

    val array = FastIndexedSeq(0,1,2)

    val region = Region()
    val rv = RegionValue(region)

    val regionValueOffsetContainingArray = ScalaToRegionValue(region, rt, array)

    println(s"regionValueOffsetContainingArray: $regionValueOffsetContainingArray")

    // we're passing 2 args; the region, and the address of the type that we pass to the region
    // return is long
    val fb = EmitFunctionBuilder[Region, Long, Long]("foo")
    val addressOfArray = fb.getArg[Long](2)
    println(s"ADDRESS OF ARRAY $addressOfArray")
    val srvb = new StagedRegionValueBuilder(fb, PInt64())

    println(s"Immediately after initialized srvb offset is ${srvb.currentOffset}")

    val condition = Code(

    )
    //generates the function
    fb.emit(Code(
      srvb.start(),
      // Works, as long as the type on srvb is PInt64()
      // Will not work otherwise
      // This should go in our pointer arrays method to generate code for storing a pointer
      srvb.addIRIntermediate(rt)(addressOfArray),
      srvb.end()
    ))

    val f = fb.result()()
    val region2 = Region();
    val rv2 = RegionValue(region2)
    val returnAddress = f(region2 , regionValueOffsetContainingArray)

    val region3 = Region();
    println(UnsafeRow.read(rt, region3, regionValueOffsetContainingArray))

//    println(s"Value read using the return address")
//    println(UnsafeRow.read(rt, region3, returnAddress))

    println(s"Value")
    println(rt.read(region2, returnAddress))
    println(s"Length ${rt.loadLength(returnAddress)}")
/*
    val rv3 = RegionValue(region4)
    val rvb = new RegionValueBuilder(region4)

    rvb.start(rt)
    rvb.startArray(3)
    for (i <- 1 to 3) {
      rvb.addInt(i)
    }
    rvb.endArray()

    rv3.setOffset(rvb.end())

    if (showRVInfo) {
      printRegion(region4, "struct with array")
      println(rv3.pretty(rt))
    }*/
//    println(s"Value 2323read using324the returned l32ong 222")

//        println(rv2.pretty(rt))

//        println("Value inserted into region")
//        rv.setOffset(regionValueOffsetContainingArray)
//        println(rv.pretty(rt))







        // this region is only for global values
//        Region.scoped { region =>
//          val rvb = new RegionValueBuilder(region)
//          val rv = RegionValue(region)
//          val f = fb.result()
//          val memoryAddressOfInsertedValue = f().apply(region, expected)
//
//          rv.setOffset(memoryAddressOfInsertedValue)

    //      rv.setOffset(memoryAddressOfInsertedValue)
    //      println("BEFORE")
    //      printRegion(region, "struct with array 1")
    //      println(rv.pretty(rt))
    //      println("AFTER")
    //      if (showRVInfo) {
    //        printRegion(region, "struct with array 1")
    //        println(rv.pretty(rt))
    //      }
    //
    //
    //      // start takes the physical type that is the "root" of the value i need
    //      rvb.start(rt)
    //      // this actually adds the data
    //      rvb.addAnnotation(rt.virtualType, expected)
    //      val addr = rvb.end()
    //      println(s"addr=$addr")
    //      // prepares the class that fb.emit stages to be executed
    //      //gives me an instance of the callable class
    //      rvb.startArray(2)
    //      for (i <- 0 to 2) {
    //        rvb.addInt(expected(i))
    //      }
    //      rvb.endArray()
    //
    //      val f = fb.result()()(r, r)
    //      // rvb.end() returns offset to the whole value
    //      val returnAddress = f(r, addr)
    //
    //      assert(UnsafeRow.read(rt, r, returnAddress) == expected)
//        }
  }
  @Test
  def testStructWithArray2a() {
    val rt = PArray(PInt32())
    val requiredRt = PArray(PInt32(), true)
    val pointerType = PInt64(true)

    val array = FastIndexedSeq(0, 1, 2)

    val region = Region()

    val regionValueOffsetContainingArray = ScalaToRegionValue(region, rt, array)

    println(s"regionValueOffsetContainingArray: $regionValueOffsetContainingArray")

    // we're passing 2 args; the region, and the address of the type that we pass to the region
    // return is long
    val fb = EmitFunctionBuilder[Region, Long, Long]("foo")

//    val rc = new EmitRegion(fb)
    val regionCode: Code[Region] = fb.getArg[Region](1)
    val addressOfArray = fb.getArg[Long](2)
    println(s"ADDRESS OF ARRAY $addressOfArray")
    val offset = fb.newField[Long]("array_offset")
    val length = fb.newField[Long]("length")

    def thing(region: Code[Region]) {
      //generates the function
      fb.emit(
        rt.fundamentalType match {
          case t: PArray => Code
            (
              length := region.allocate(t.contentsAlignment, t.contentsByteSize(t.loadLength(addressOfArray.load())))
              //      offset := regionCode.asInstanceOf[Code[Region]].allocate(rt.contentsAlignment, rt.contentsByteSize(rt.loadLength(regionValueOffsetContainingArray)))

              //      Code(Region.copyFrom(regionValueOffsetContainingArray, offset, length))
              //      offset := region.allocate(pointerType.alignment, pointerType.byteSize),
              //      Region.storeAddress(offset, addressOfArray)
              )
        })
    }

    println("Before result")
    val f = fb.result()()
    println("Past fb.result()()")
    val region2 = Region();

    val returnAddress = f(region, regionValueOffsetContainingArray)
    println("PAST retrunAddredss")
    println(s"RETURN ADDRESS $returnAddress")

    val theOriginalAddress = Region.loadLong(returnAddress)
    println(s"The value inserted at the returnAddress: $theOriginalAddress")

    val addr2 = Region.loadAddress(regionValueOffsetContainingArray)
    println(s"THE LONG VALUE WE INSERTED INTO THE ORIG regionValueOffsetContainingArray: $addr2")
    // Proof that the region doesn't have its own memory offsets.
    // The return address is into raw memory
    val region3 = Region();
    println(s"Value read in uninitialized region")
    println(UnsafeRow.read(rt, region3, regionValueOffsetContainingArray))

    //    println(s"Value read using the return address")
    //    println(UnsafeRow.read(rt, region3, returnAddress))

    println(s"Value read ddd the returned l32ong 222")
    println(UnsafeRow.read(rt, region3, theOriginalAddress))
    println("Trying to pretty print")

    val region4 = Region()
    val rv3 = RegionValue(region4)
    rv3.setOffset(theOriginalAddress)
    println(rv3.pretty(rt))
  }

  def testStructWithArray3() {
    val rt = PArray(PInt32())
    val requiredRt = PArray(PInt32(), true)

    val array = FastIndexedSeq(0, 1, 2)

    val region = Region()
    val rv = RegionValue(region)

    val regionValueOffsetContainingArray = ScalaToRegionValue(region, rt, array)

    println(s"regionValueOffsetContainingArray: $regionValueOffsetContainingArray")

    // we're passing 2 args; the region, and the address of the type that we pass to the region
    // return is long
    val fb = EmitFunctionBuilder[Region, Long, Long]("foo")
    val addressOfArray = fb.getArg[Long](2)
    println(s"ADDRESS OF ARRAY $addressOfArray")
    val srvb = new StagedRegionValueBuilder(fb, rt)
    val length = fb.newField[Long]("array_length")
    println(s"Immediately after initialized srvb offset is ${ srvb.currentOffset }")
  }

//  @Test def testShallowArrayCopy() {
//    val t = PArray(PArray(PInt32()))
//    val a = FastIndexedSeq(FastIndexedSeq(1,2,3), FastIndexedSeq(4,5,6))
//
//    println(s"A! $a")
//    assert(t.virtualType.typeCheck(a))
//
//      val copy = Region.scoped { region =>
//        val copyOff = Region.scoped { srcRegion =>
//          val src = ScalaToRegionValue(srcRegion, t, a)
//
//          val fb = EmitFunctionBuilder[Region, Long, Long]("deep_copy")
//          fb.emit(
//            t.upcastFromOffset(
//              EmitRegion.default(fb.apply_method),
//              t,
//              fb.getArg[Long](2).load()))
//          val copyF = fb.resultWithIndex()(0, region)
//          val newOff = copyF(region, src)
//
//          // This isn't needed for this test to work
//          //clear old stuff
//          // actually without this  will pass always, even if the code didn't generate the right array
////          val len = srcRegion.allocate(0) - src
////          Region.storeBytes(src, Array.fill(len.toInt)(0.toByte))
//          newOff
//        }
//        // SafeIndexedSeq makes another copy
//        // UnsafeIndexedSeq works, but it gives back a [1,2,3],[4,5,6]
//        // while SafeIndexedSeq gives back WrappedArray(WrappedArray(1,2,3), WrappedArray(4,5,6))
////        new UnsafeIndexedSeq(t, region, copyOff)
//        SafeIndexedSeq(t, region, copyOff)
//      }
//
//    println(s"COPY IS $copy")
//
//    assert(copy == a)
//  }

//  @Test def testDeepArrayCopy() {
//    val sourceType = PArray(PArray(PInt32(true)))
//    val sourceValue = FastIndexedSeq(FastIndexedSeq(1,2,3), FastIndexedSeq(4,5,6))
//
//    val destType = PArray(PArray(PInt32()))
//
//
//    println(s"A! $sourceValue")
//    assert(sourceType.virtualType.typeCheck(sourceValue))
//
//    val copy = Region.scoped { region =>
//      val copyOff = Region.scoped { srcRegion =>
//        val src = ScalaToRegionValue(srcRegion, sourceType, sourceValue)
//
//        val fb = EmitFunctionBuilder[Region, Long, Long]("deep_copy")
//        fb.emit(
//          sourceType.upcastFromOffset(
//            EmitRegion.default(fb.apply_method),
//            destType,
//            fb.getArg[Long](2).load(),false))
//        val copyF = fb.resultWithIndex()(0, region)
//        val newOff = copyF(region, src)
//
//        // This isn't needed for this test to work
//        //clear old stuff
//        // actually without this  will pass always, even if the code didn't generate the right arra
//        val len = srcRegion.allocate(0) - src
//        Region.storeBytes(src, Array.fill(len.toInt)(0.toByte))
//        newOff
//      }
//      // SafeIndexedSeq makes another copy
//      // UnsafeIndexedSeq works, but it gives back a [1,2,3],[4,5,6]
//      // while SafeIndexedSeq gives back WrappedArray(WrappedArray(1,2,3), WrappedArray(4,5,6))
//      new UnsafeIndexedSeq(destType, region, copyOff)
//      //        SafeIndexedSeq(t, region, copyOff)
//    }
//
//    println(s"COPY IS $copy")
//
//    assert(copy == sourceValue)
//  }

  def test(): AsmFunction0[Int] = {
    val fb = functionBuilder[Int]
    val l = fb.newLocal[Int]
    fb.emit(Code(l:=1, _return(l)))
    val f = fb.result()()

    f
//    val fb = EmitFunctionBuilder[Long]("blah")
//    val test = fb.newField[String]("field")
//    fb.emit(
//      Code(
//        test := "HELLO WORLD",
//        Code._println(test)
//      )
//    )
//    fb.result()()
  }


  @Test def testDeepArrayUpcastFlatElementNotRequired() {
    val sourceType = PArray(PArray(PArray(PInt64(true), true), true), true)
    val destType = PArray(PArray(PArray(PInt64(false))))
    val sourceValue = FastIndexedSeq(FastIndexedSeq(FastIndexedSeq(1L,2L,0L,3L,4L)), FastIndexedSeq(FastIndexedSeq(20L,21L,31L,41L)), FastIndexedSeq(FastIndexedSeq(0L,7L,9L,2L)))

    val region = Region()
    val srcRegion = Region()

    val src = ScalaToRegionValue(srcRegion, sourceType, sourceValue)

    val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
    val codeRegion = fb.getArg[Region](1).load()
    val value = fb.getArg[Long](2)

    fb.emit(destType.copyDataOfDifferentType(fb, codeRegion, sourceType, value))

    val f = fb.result()()
    val copyOff = f(region,src)

    val copy = SafeIndexedSeq(destType, region, copyOff)

    assert(copy == sourceValue)
  }

  @Test def testSimpleArrayCopy() {
    val sourceType = PArray(PInt64(true),true)
    val destType = PArray(PInt64())
    val sourceValue = FastIndexedSeq(1L,2L,0L,3L,4L)

    val region = Region()
    val srcRegion = Region()

    val src = ScalaToRegionValue(srcRegion, sourceType, sourceValue)

    val fb = EmitFunctionBuilder[Region, Long, Long]("not_empty")
    val codeRegion = fb.getArg[Region](1).load()
    val value = fb.getArg[Long](2)

    fb.emit(destType.copyDataOfDifferentType(fb, codeRegion, sourceType, value))

    val f = fb.result()()
    val copyOff = f(region,src)

    val copy = SafeIndexedSeq(destType, region, copyOff)

    assert(copy == sourceValue)
  }

  @Test def testDeepCopy() {
    val g = Type.genStruct
      .flatMap(t => Gen.zip(Gen.const(t), t.genValue))
      .filter { case (t, a) => a != null }
      .map { case (t, a) => (PType.canonical(t).asInstanceOf[PStruct], a) }
    println(g)
    val p = Prop.forAll(g) ( v => {
      println(s"G :$g")
      // v is an instance of Gen class
      v match {
        case (t, a) =>
          println(s"A! $a, ${a.getClass().toString()}")
          assert(t.virtualType.typeCheck(a))
          val copy = Region.scoped { region =>
            val copyOff = Region.scoped { srcRegion =>
              val src = ScalaToRegionValue(srcRegion, t, a)

              val fb = EmitFunctionBuilder[Region, Long, Long]("deep_copy")
              fb.emit(
                StagedRegionValueBuilder.deepCopyFromOffset(
                  EmitRegion.default(fb.apply_method),
                  t,
                  fb.getArg[Long](2).load()))
              val copyF = fb.resultWithIndex()(0, region)
              val newOff = copyF(region, src)


              //clear old stuff
              val len = srcRegion.allocate(0) - src
              Region.storeBytes(src, Array.fill(len.toInt)(0.toByte))
              newOff
            }
            SafeRow(t, region, copyOff)
          }

          println(s"COPY IS $copy")
          copy == a
      }
    })
    p.check()
  }
}
