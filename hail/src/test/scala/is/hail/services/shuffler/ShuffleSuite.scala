package is.hail.shuffler

import is.hail.asm4s._
import org.apache.log4j.Logger;
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.types.physical._
import is.hail.types.encoded._
import is.hail.expr.ir.ArrayZipBehavior._
import is.hail.shuffler.server._
import is.hail.shuffler.ShufflerTestUtils._
import is.hail.io._
import is.hail.utils._
import is.hail._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.util.Random
import scala.language.implicitConversions

class ShuffleSuite extends HailSuite {
  private[this] val log = Logger.getLogger(getClass.getName())

  private[this] def arrayOfUnsafeRow(elementPType: PStruct, array: Array[Long]): Array[UnsafeRow] =
    array.map(new UnsafeRow(elementPType, null, _)).toArray

  @Test def testShuffle() {
    var server = new ShuffleServer(sslContext(
      "src/test/resources/non-secret-key-and-trust-stores/server-keystore.p12",
      "hailhail",
      "PKCS12",
      "src/test/resources/non-secret-key-and-trust-stores/server-truststore.p12",
      "hailhail",
      "JKS"
    ),
      8080)
    server.serveInBackground()
    try {
      val rowPType = PCanonicalStruct("x" -> PInt32())
      val rowType = rowPType.virtualType.asInstanceOf[TStruct]
      val key = Array("x")
      val keyFields = key.map(x => SortField(x, Ascending)).toArray
      val keyType = rowType.typeAfterSelectNames(key)
      val keyPType = rowPType.selectFields(key)
      val rowEType = EType.defaultFromPType(rowPType).asInstanceOf[EBaseStruct]
      val keyEType = EType.defaultFromPType(keyPType).asInstanceOf[EBaseStruct]
      val shuffleType = TShuffle(keyFields, rowType, rowEType, keyEType)
      using(new ShuffleClient(
        shuffleType,
        sslContext(
          "src/test/resources/non-secret-key-and-trust-stores/client-keystore.p12",
          "hailhail",
          "PKCS12",
          "src/test/resources/non-secret-key-and-trust-stores/client-truststore.p12",
          "hailhail",
          "JKS"),
        "localhost",
        8080)) { c =>
        val rowDecodedPType = c.codecs.rowDecodedPType

        val values = new ArrayBuilder[Long]()
        Region.scoped { region =>
          val rvb = new RegionValueBuilder(region)
          val nElements = 1000000
          val nPartitions = 100
          val shuffled = Random.shuffle((0 until nElements).toIndexedSeq).toArray
          var i = 0
          while (i < nElements) {
            rvb.start(rowPType)
            rvb.startStruct()
            rvb.addInt(shuffled(i))
            rvb.endStruct()
            values += rvb.end()
            i += 1
          }

          c.start()
          c.put(values.result())

          val partitionBounds = arrayOfUnsafeRow(
            c.codecs.keyDecodedPType,
            c.partitionBounds(region, nPartitions))
          assert(partitionBounds.length == nPartitions + 1)
          val keyOrd = c.codecs.keyDecodedPType.unsafeOrdering
          assert(nPartitions < nElements)
          assertStrictlyIncreasingPrefix(
            keyOrd, partitionBounds, partitionBounds.length)

          val left = struct(0)
          val right = struct(nElements)

          val fromOneQuery = arrayOfUnsafeRow(
            rowDecodedPType,
            c.get(region, left, true, right, false))

          i = 0
          val ab = new ArrayBuilder[Long]()
          while (i < nPartitions) {
            ab ++= c.get(region,
              partitionBounds(i).offset, true,
              partitionBounds(i + 1).offset, i == nPartitions - 1)
            i += 1
          }
          val fromPartitionedQueries = arrayOfUnsafeRow(
            rowDecodedPType,
            ab.result())

          i = 0
          def assertFirstFieldEqualsIndex(rows: IndexedSeq[UnsafeRow], i: Int): Unit = {
            assert(!rows(i).isNullAt(0),
              s"""first field is undefined ${rows(i)}
               |Context: ${rows.slice(i-3, i+3)}.
               |Length: ${rows.length}""".stripMargin)
            assert(rows(i).getInt(0) == i,
              s"""first field should be ${i}: ${rows(i)}.
               |Context: ${rows.slice(i-3, i+3)}.
               |Length: ${rows.length}""".stripMargin)
          }
          while (i < nElements) {
            assertFirstFieldEqualsIndex(fromOneQuery, i)
            assertFirstFieldEqualsIndex(fromPartitionedQueries, i)
            i += 1
          }
          assert(fromOneQuery.length == nElements)
          assert(fromPartitionedQueries.length == nElements)
        }
        c.stop()
      }
    } finally {
      if (server != null) {
        server.stop()
      }
    }
  }

  @Test def testShuffleIR() {
    var server = new ShuffleServer(sslContext(
      "src/test/resources/non-secret-key-and-trust-stores/server-keystore.p12",
      "hailhail",
      "PKCS12",
      "src/test/resources/non-secret-key-and-trust-stores/server-truststore.p12",
      "hailhail",
      "JKS"
    ),
      8080)
    server.serveInBackground()
    try {
      val rowPType = PCanonicalStruct("x" -> PInt32())
      val rowType = rowPType.virtualType.asInstanceOf[TStruct]
      val key = Array("x")
      val keyFields = key.map(x => SortField(x, Ascending)).toArray
      val keyType = rowType.typeAfterSelectNames(key)
      val keyPType = rowPType.selectFields(key)
      val rowEType = EType.defaultFromPType(rowPType).asInstanceOf[EBaseStruct]
      val keyEType = EType.defaultFromPType(keyPType).asInstanceOf[EBaseStruct]
      val shuffleType = TShuffle(keyFields, rowType, rowEType, keyEType)
      val nElements = 1000000
      val partSize = 10000
      val nParts = nElements / partSize
      assert(nElements % partSize == 0)
      val keyOrd = keyPType.unsafeOrdering
      val x =
        ShuffleWith(keyFields, rowType, rowEType, keyEType,
          "id",
          Let(
            "input",
            ToArray(
              StreamMap(
                ToStream(Literal(TArray(TInt32), Random.shuffle((0 until nElements).toIndexedSeq).toFastIndexedSeq)),
                "x",
                MakeStruct(Seq("x" -> Ref("x", TInt32))))),
            ToArray(StreamMap(
              StreamRange(I32(0), I32(nParts), I32(1)),
              "partitionId",
              ShuffleWrite(
                Ref("id", shuffleType),
                ToStream(
                  invoke(
                    "slice",
                    TArray(keyType),
                    Ref("input", TArray(keyType)),
                    ApplyBinaryPrimOp(
                      Multiply(),
                      Ref("partitionId", TInt32),
                      I32(partSize)),
                    ApplyBinaryPrimOp(
                      Multiply(),
                      ApplyBinaryPrimOp(
                        Add(),
                        Ref("partitionId", TInt32),
                        I32(1)),
                      I32(partSize)))))))),
          Let(
            "boundaries",
            ToArray(ShufflePartitionBounds(Ref("id", shuffleType), I32(nParts))),
            Let(
              "nPartitions",
              ApplyBinaryPrimOp(
                Subtract(),
                ArrayLen(Ref("boundaries", TArray(keyType))),
                I32(1)),
              MakeTuple(FastSeq(
                (0, Ref("boundaries", TArray(keyType))),
                (1,
                  ToArray(
                    StreamFlatMap(
                      StreamZip(
                        FastSeq(
                          ToStream(Ref("boundaries", TArray(keyType))),
                          StreamDrop(ToStream(Ref("boundaries", TArray(keyType))), I32(1)),
                          StreamRange(
                            I32(0),
                            Ref("nPartitions", TInt32),
                            I32(1))),
                        FastSeq("l", "r", "index"),
                        ToArray(ShuffleRead(
                          Ref("id", shuffleType),
                          invoke(
                            "Interval",
                            TInterval(keyType),
                            Ref("l", keyType),
                            Ref("r", keyType),
                            True(),
                            ApplyComparisonOp(
                              EQ(TInt32, TInt32),
                              Ref("index", TInt32),
                              ApplyBinaryPrimOp(
                                Subtract(),
                                Ref("nPartitions", TInt32),
                                I32(1)))))),
                        ArrayZipBehavior.TakeMinLength),
                      "a",
                      ToStream(Ref("a", TArray(rowType)))))))))))
      ExecuteContext.scoped() { (ctx: ExecuteContext) =>
        val region = ctx.r
        val (pairPType: PBaseStruct, f) = Compile[AsmFunction1RegionLong](ctx,
          FastIndexedSeq[(String, PType)](),
          FastIndexedSeq[TypeInfo[_]](classInfo[Region]), LongInfo,
          x,
          optimize = true)
        val partitionBoundsPType = pairPType
          .asInstanceOf[PBaseStruct].types(0)
        val rowArrayPType = pairPType
          .asInstanceOf[PBaseStruct].types(1)
          .asInstanceOf[PArray]
        val elementPType = rowArrayPType.elementType
          .asInstanceOf[PBaseStruct]
        val pairOff = f(0, ctx.r)(ctx.r)
        val pair = new UnsafeRow(pairPType, null, pairOff)
        val partitionBounds = pair.get(0).asInstanceOf[IndexedSeq[UnsafeRow]]
        assert(nParts > 0)
        assert(partitionBounds.length == nParts + 1)
        assertStrictlyIncreasingPrefix(
          keyOrd, partitionBounds.toArray, partitionBounds.length)

        val rowArray = pair.get(1).asInstanceOf[IndexedSeq[UnsafeRow]]
        assert(rowArray.length == nElements)

        var i = 0
        while (i < nElements) {
          assert(i < rowArray.length, s"$i ${rowArray.length}")
          assert(!rowArray(i).isNullAt(0),
            s"first field is undefined ${rowArray(i)}")
          assert(rowArray(i).getInt(0) == i,
            s"first field should be ${i}: ${rowArray(i)}. Context: ${rowArray.slice(i-3, i+3)}. Length: ${rowArray.length}")
          i += 1
        }
      }
    } finally {
      if (server != null) {
        server.stop()
      }
    }
  }
}
