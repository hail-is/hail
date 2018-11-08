package is.hail.nativecode

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.cxx._
import is.hail.expr.types._
import is.hail.io._
import is.hail.rvd.RVDContext
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class RegionValueIteratorSuite extends SparkSuite {

  def getData(): (BufferSpec, TTuple, IndexedSeq[Row]) = {
    val spec = new LEB128BufferSpec(
      new BlockingBufferSpec(32,
        new LZ4BlockBufferSpec(32,
          new StreamBlockBufferSpec)))
    val t = TTuple(TInterval(TStruct("x"->TSet(TInt32()))))

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
    val decMod = PackDecoder.buildModule(t.physicalType, t.physicalType, spec)

    val tub = new TranslationUnitBuilder
    val encClass = PackEncoder(t.physicalType, spec, tub)
    tub += encClass

    tub.include("<jni.h>")
    tub.include("hail/PartitionIterators.h")
    val partitionFB = FunctionBuilder("partition_f", Array("NativeStatus*" -> "st", "long" -> "objects"), "long")
    val up = Variable("up", "UpcallEnv")
    val encoder = Variable("encoder", encClass.name, s"std::make_shared<OutputStream>($up, reinterpret_cast<ObjectArray * >(${ partitionFB.getArg(1) })->at(1))")
    val it = Variable("it", "JavaRVIterator", s"std::make_shared<JavaIteratorWrapper>($up, reinterpret_cast<ObjectArray * >(${ partitionFB.getArg(1) })->at(0))")

    partitionFB += up.define
    partitionFB += encoder.define
    partitionFB += it.define
    partitionFB +=
      s"""
         |for(++$it; $it != $it.end(); ++$it) {
         |  $encoder.encode_row(${ partitionFB.getArg(0) }, *$it);
         |}
         |$encoder.flush(${ partitionFB.getArg(0) });
         |return 0;
       """.stripMargin

    tub += partitionFB.result()

    val mod = tub.result().build("-O2 -llz4")
    val key = mod.getKey
    val bin = mod.getBinary

    val result = hc.sc.parallelize(a, 2).mapPartitions { rowsIt =>
      val rows = rowsIt.toIndexedSeq
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        val it = new RegionValueIterator(rows.toIterator.map { r =>
          rvb.start(t.physicalType)
          rvb.addAnnotation(t, r)
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
          SafeRow(t.asInstanceOf[TBaseStruct].physicalType, region, decoder.readRegionValue(region))
        }.toIterator
      }
    }.collect()

    assert(result sameElements a)
  }

  @Test def testCXXRegionValueIterator(): Unit = {
    val (spec, t, a) = getData()

    val encMod = PackEncoder.buildModule(t.physicalType, spec)

    val makeIt = CXXRegionValueIterator { tub: TranslationUnitBuilder =>
      val decClass = PackDecoder(t.physicalType, t.physicalType, spec, tub)
      tub += decClass

      val cb = new ClassBuilder("CXXIterator", "NativeObj")
      val it = Variable("it", s"std::shared_ptr<ReadIterator<${ decClass.name }>>")
      val region = Variable("region", s"Region *")
      val st = Variable("st", s"NativeStatus *")
      val value = Variable("v", s"char *")
      cb.addPrivate(it)
      cb.addConstructor(
        s"""${cb.name}(jobject is, Region * reg, NativeStatus * st) :
           |$it(std::make_shared<ReadIterator<${ decClass.name }>>(std::make_shared<${ decClass.name }>(std::make_shared<InputStream>(UpcallEnv(), is)), reg, st)) { }
         """.stripMargin)

      cb += new Function(s"${ cb.name }&", "operator++", Array(), s"++(*$it); return *this;")
      cb += new Function("char *", "operator*", Array(), s"return *(*$it);")
      cb.result()
    }

    val encoded = hc.sc.parallelize(a, 2).mapPartitions { case (rowsIt) =>
      val rows = rowsIt.toIndexedSeq
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        val baos = new ByteArrayOutputStream()
        val encoder = new NativePackEncoder(baos, encMod)
        rows.foreach { r =>
          rvb.start(t.physicalType)
          rvb.addAnnotation(t, r)
          encoder.writeByte(1)
          encoder.writeRegionValue(region, rvb.end())
        }
        encoder.writeByte(0)
        encoder.flush()
        Iterator.single(baos.toByteArray)
      }
    }

    val result = ContextRDD.weaken[RVDContext](encoded).cmapPartitions { case (ctx, it) =>
      val bais = new ByteArrayInputStream(it.next())
      makeIt(ctx.region, bais)
    }.mapPartitions { it =>
      it.map { rv =>
        SafeRow(t.asInstanceOf[TBaseStruct].physicalType, rv)
      }
    }.collect()

    assert(result sameElements a)
  }
}
