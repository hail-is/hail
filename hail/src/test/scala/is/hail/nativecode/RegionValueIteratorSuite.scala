package is.hail.nativecode

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailSuite
import is.hail.annotations._
import is.hail.cxx._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io._
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class RegionValueIteratorSuite extends HailSuite {

  def getData(): (BufferSpec, PTuple, IndexedSeq[Row]) = {
    val spec = new LEB128BufferSpec(
      new BlockingBufferSpec(32,
        new LZ4BlockBufferSpec(32,
          new StreamBlockBufferSpec)))
    val t = PTuple(PInterval(PStruct("x" -> PSet(PInt32()))))

    val a = FastIndexedSeq(
      Row(Interval(Row(Set(-1478292367)), Row(Set(2084728308)), true, true)),
      Row(Interval(Row(Set(-1)), Row(Set(2346211)), true, false)),
      Row(Interval(Row(Set(25)), Row(Set(70)), false, true)),
      Row(Interval(Row(Set(50)), Row(Set(48)), false, false)))

    (spec, t, a)
  }

  @Test def testScalaRegionValueIterator(): Unit = {
    hc
    val (spec, t, a) = getData()
    val decMod = PackDecoder.buildModule(t, t, spec)

    val tub = new TranslationUnitBuilder
    val encClass = PackEncoder(t, spec, tub)

    tub.include("<jni.h>")
    tub.include("hail/PartitionIterators.h")
    val partitionFB = tub.buildFunction("partition_f", Array("NativeStatus*" -> "st", "long" -> "objects"), "long")
    val up = tub.variable("up", "UpcallEnv")
    val encoder = tub.variable("encoder", encClass.name, s"std::make_shared<OutputStream>($up, reinterpret_cast<ObjectArray * >(${ partitionFB.getArg(1) })->at(1))")
    val jit = tub.variable("jit", "JavaIteratorObject", s"JavaIteratorObject($up, reinterpret_cast<ObjectArray * >(${ partitionFB.getArg(1) })->at(0))")

    partitionFB +=
      s"""
         |${ up.define }
         |${ encoder.define }
         |${ jit.define }
         |for(auto it : $jit) {
         |  $encoder.encode_row(it);
         |}
         |$encoder.flush();
         |return 0;
       """.stripMargin
    partitionFB.end()

    val mod = tub.end().build("-O2 -llz4")
    val key = mod.getKey
    val bin = mod.getBinary

    val result = hc.sc.parallelize(a, 2).mapPartitions { rowsIt =>
      val rows = rowsIt.toFastIndexedSeq
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        val it = new RegionValueIterator(rows.toIterator.map { r =>
          rvb.start(t)
          rvb.addAnnotation(t.virtualType, r)
          RegionValue(region, rvb.end())
        })

        val baos = new ByteArrayOutputStream()
        val obj = new ObjectArray(it, baos)

        val st = new NativeStatus()
        val mod = new NativeModule(key, bin)
        val f = mod.findLongFuncL1(st, "partition_f")
        f(st, obj.get())

        val bais = new ByteArrayInputStream(baos.toByteArray)
        val decoder = new NativePackDecoder(bais, decMod)

        rows.map { r =>
          SafeRow(t, region, decoder.readRegionValue(region))
        }.toIterator
      }
    }.collect()

    assert(result sameElements a)
  }

  @Test def testCXXRegionValueIterator(): Unit = {
    val (spec, t, a) = getData()

    val encMod = PackEncoder.buildModule(t, spec)
    val tub = new TranslationUnitBuilder()
    val decClass = PackDecoder(t, t, "InputStream", spec, tub)
    tub.include("hail/PartitionIterators.h")
    tub.include("hail/ObjectArray.h")
    val makeItF = tub.buildFunction("make_iterator", Array("NativeStatus *"->"st", "long" -> "reg", "long" -> "obj"), "NativeObjPtr")
    val itType = s"Reader<${ decClass.name }>"
    val is = s"std::make_shared<InputStream>(UpcallEnv(), reinterpret_cast<ObjectArray *>(${ makeItF.getArg(2) })->at(0))"
    makeItF += s"return std::make_shared<ScalaStagingIterator<$itType>>(${ decClass.name }($is), reinterpret_cast<ScalaRegion *>(${ makeItF.getArg(1) }));"
    makeItF.end()

    val modToPtr = { (mod: NativeModule, region: Region, obj: ObjectArray) =>
      val st = new NativeStatus()
      val ptrF = mod.findPtrFuncL2(st, "make_iterator")
      scala.Predef.assert(st.ok, st.toString())
      val ptr = new NativePtr(ptrF, st, region.get(), obj.get())
      scala.Predef.assert(st.ok, st.toString())
      ptrF.close()
      st.close()
      obj.close()
      ptr
    }

    val makeIt = CXXRegionValueIterator(s"ScalaStagingIterator<$itType>", tub, modToPtr)

    val encoded = hc.sc.parallelize(a, 2).mapPartitions { case (rowsIt) =>
      val rows = rowsIt.toFastIndexedSeq
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        val baos = new ByteArrayOutputStream()
        val encoder = new NativePackEncoder(baos, encMod)
        rows.foreach { r =>
          rvb.start(t)
          rvb.addAnnotation(t.virtualType, r)
          encoder.writeByte(1)
          encoder.writeRegionValue(region, rvb.end())
        }
        encoder.writeByte(0)
        encoder.flush()
        Iterator.single(baos.toByteArray)
      }
    }

    val result = ContextRDD.weaken[RVDContext](encoded).cmapPartitions { case (ctx, it) =>
      makeIt(ctx.region, new ObjectArray(new ByteArrayInputStream(it.next())))
    }.map(SafeRow(t, _)).collect()

    assert(result sameElements a)
  }
}
