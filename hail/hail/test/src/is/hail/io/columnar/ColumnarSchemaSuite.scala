package is.hail.io.columnar

import is.hail.TestUtils._
import is.hail.collection.FastSeq
import is.hail.expr.Nat
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.variant.ReferenceGenome

import scala.jdk.CollectionConverters._

import java.io.ByteArrayInputStream
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.Channels
import java.util.Base64

import org.apache.arrow.vector.ipc.ReadChannel
import org.apache.arrow.vector.ipc.message.MessageSerializer
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.parquet.schema.MessageTypeParser
import org.junit.jupiter.api.Test

class ColumnarSchemaSuite {

  private def leaf(name: String, t: ArrowType, nullable: Boolean = true): Field =
    new Field(name, new FieldType(nullable, t, null), FastSeq.empty[Field].asJava)

  private def node(name: String, t: ArrowType, children: Field*): Field =
    new Field(name, new FieldType(true, t, null), children.asJava)

  private val Int32 = new ArrowType.Int(32, true)
  private val Int64 = new ArrowType.Int(64, true)

  private def fieldIds(f: Field): Seq[(String, Option[String])] =
    (f.getName -> Option(f.getMetadata.get(ColumnarSchema.FieldIdKey))) +:
      f.getChildren.asScala.toSeq.flatMap(fieldIds)

  @Test def testFieldIdsArePreOrderAndSkipMapEntries(): Unit = {
    val schema = new Schema(FastSeq(
      leaf("a", Int32),
      node(
        "b",
        ArrowType.Struct.INSTANCE,
        leaf("c", ArrowType.Utf8.INSTANCE),
        node("d", ArrowType.List.INSTANCE, leaf("element", Int64)),
      ),
      node(
        "e",
        new ArrowType.Map(true),
        new Field(
          "entries",
          new FieldType(false, ArrowType.Struct.INSTANCE, null),
          FastSeq(leaf("key", ArrowType.Utf8.INSTANCE, nullable = false), leaf("value", Int32))
            .asJava,
        ),
      ),
    ).asJava)

    val withIds = ColumnarSchema.withFieldIds(schema)

    assertEq(
      withIds.getFields.asScala.toSeq.flatMap(fieldIds),
      Seq(
        "a" -> Some("1"),
        "b" -> Some("2"),
        "c" -> Some("3"),
        "d" -> Some("4"),
        "element" -> Some("5"),
        "e" -> Some("6"),
        "entries" -> None,
        "key" -> Some("7"),
        "value" -> Some("8"),
      ),
    )
  }

  @Test def testParquetSchemaUsesStandardEncodingsAndFieldIds(): Unit = {
    val schema = ColumnarSchema.withFieldIds(new Schema(FastSeq(
      leaf("__row_id", Int64, nullable = false),
      leaf("a", ArrowType.Utf8.INSTANCE),
      node("b", ArrowType.List.INSTANCE, leaf("element", Int32, nullable = false)),
      new Field(
        "c",
        new FieldType(false, ArrowType.Struct.INSTANCE, null),
        FastSeq(
          leaf("d", ArrowType.Bool.INSTANCE),
          leaf("e", ArrowType.Binary.INSTANCE, nullable = false),
          leaf("f", new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
          leaf("g", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
        ).asJava,
      ),
      leaf("h", Int32),
    ).asJava))

    val expected = MessageTypeParser.parseMessageType(
      """message schema {
        |  required int64 __row_id = 1;
        |  optional binary a (STRING) = 2;
        |  optional group b (LIST) = 3 {
        |    repeated group list {
        |      required int32 element = 4;
        |    }
        |  }
        |  required group c = 5 {
        |    optional boolean d = 6;
        |    required binary e = 7;
        |    optional float f = 8;
        |    optional double g = 9;
        |  }
        |  optional int32 h = 10;
        |}
        |""".stripMargin
    )

    val actual = ColumnarSchema.toParquet(schema)
    assertEq(actual.toString, expected.toString)
    assertEq(actual, expected)
  }

  @Test def testParquetSchemaOfMapsTensorsAndExtensionTypes(): Unit = {
    val rg = ReferenceGenome("small", FastSeq("1", "2"), Map("1" -> 10, "2" -> 10))
    val schema = ColumnarSchema.fileSchema(
      PCanonicalStruct(
        true,
        "d" -> PCanonicalDict(PCanonicalString(), PInt32(), false),
        "nd" -> PCanonicalNDArray(PFloat64(true), 2, true),
        "l" -> PCanonicalLocus(rg, true),
        "c" -> PCanonicalCall(false),
        "s" -> PCanonicalSet(PInt32(true), false),
        "i" -> PCanonicalInterval(PInt32(true), true),
        "t" -> PCanonicalTuple(false, PInt32(true), PCanonicalString(false)),
        "e" -> PCanonicalStruct(false),
        "et" -> PCanonicalTuple(true),
      ),
      None,
    )

    val expected = MessageTypeParser.parseMessageType(
      """message schema {
        |  optional group d (MAP) = 1 {
        |    repeated group key_value {
        |      required binary key (STRING) = 2;
        |      optional int32 value = 3;
        |    }
        |  }
        |  required group nd = 4 {
        |    required group data (LIST) = 5 {
        |      repeated group list {
        |        required double element = 6;
        |      }
        |    }
        |    optional group shape (LIST) = 7 {
        |      repeated group list {
        |        required int32 element = 8;
        |      }
        |    }
        |  }
        |  required int64 l = 9;
        |  optional int32 c = 10;
        |  optional group s (LIST) = 11 {
        |    repeated group list {
        |      required int32 element = 12;
        |    }
        |  }
        |  required group i = 13 {
        |    required int32 start = 14;
        |    required int32 end = 15;
        |    required boolean includes_start = 16;
        |    required boolean includes_end = 17;
        |  }
        |  optional group t = 18 {
        |    required int32 0 = 19;
        |    optional binary 1 (STRING) = 20;
        |  }
        |  optional boolean e = 21;
        |  required boolean et = 22;
        |}
        |""".stripMargin
    )

    val actual = ColumnarSchema.toParquet(schema)
    assertEq(actual.toString, expected.toString)
    assertEq(actual, expected)
  }

  @Test def testEveryTypeButRngStateAndNonNumericNDArraysIsSupported(): Unit = {
    ColumnarSchema.checkSupported(
      "row",
      TStruct(
        "d" -> TDict(TString, TInt32),
        "nd" -> TNDArray(TFloat64, Nat(2)),
        "ndb" -> TNDArray(TBoolean, Nat(1)),
        "l" -> TLocus(ReferenceGenome.builtinReferences()("GRCh37")),
        "c" -> TCall,
        "s" -> TSet(TInt32),
        "i" -> TInterval(TLocus(ReferenceGenome.builtinReferences()("GRCh38"))),
        "t" -> TTuple(TInt32, TString),
        "e" -> TStruct.empty,
        "et" -> TTuple(),
        "a" -> TArray(TStruct("x" -> TStruct.empty)),
      ),
    )
  }

  @Test def testRngStateIsRejectedWithItsPath(): Unit =
    interceptFatal("global field 'a\\[\\]\\.rng' has type rng_state, which can't be stored") {
      ColumnarSchema.checkSupported(
        "global",
        TStruct("a" -> TArray(TStruct("x" -> TInt32, "rng" -> TRNGState))),
      )
    }

  @Test def testNDArrayOfNonNumericElementsIsRejectedWithItsPath(): Unit =
    FastSeq(TStruct("x" -> TInt32), TTuple(TInt32), TStruct.empty, TInterval(TInt32), TCall)
      .foreach { elementType =>
        interceptFatal(
          "row field 'a\\[\\]\\.nd' has type .*, which can't be stored: an ndarray's elements"
        ) {
          ColumnarSchema.checkSupported(
            "row",
            TStruct("a" -> TArray(TStruct("nd" -> TNDArray(elementType, Nat(1))))),
          )
        }
      }

  @Test def testOnlyNonBuiltinReferenceGenomesAreListedByName(): Unit = {
    val grch37 = ReferenceGenome.builtinReferences()("GRCh37")
    val custom = ReferenceGenome("custom", FastSeq("1", "2"), Map("1" -> 10, "2" -> 10))
    assertEq(
      ColumnarSchema.nonBuiltinReferenceGenomes(
        TStruct("l" -> TLocus(grch37), "i" -> TArray(TInterval(TLocus(custom)))),
        TStruct("nd" -> TNDArray(TFloat64, Nat(1)), "l" -> TLocus(custom)),
      ),
      FastSeq(custom),
    )
  }

  @Test def testANonBuiltinReferenceGenomeWithABuiltinNameIsRejected(): Unit = {
    val impostor = ReferenceGenome("GRCh37", FastSeq("1"), Map("1" -> 10))
    interceptFatal("reference genome 'GRCh37' has the name of a builtin reference genome") {
      ColumnarSchema.nonBuiltinReferenceGenomes(TStruct("l" -> TLocus(impostor)))
    }
  }

  @Test def testSchemaIsAnEncapsulatedIpcMessageInBase64(): Unit = {
    val schema = ColumnarSchema.withFieldIds(new Schema(FastSeq(
      leaf("a", Int32, nullable = false),
      node("b", ArrowType.List.INSTANCE, leaf("element", ArrowType.Utf8.INSTANCE)),
    ).asJava))

    val bytes = Base64.getDecoder.decode(ColumnarSchema.serialize(schema))
    val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    assertEq(header.getInt(0), 0xffffffff)
    val metadataLength = header.getInt(4)
    assertEq(bytes.length, 8 + metadataLength)
    assertEq(bytes.length % 8, 0)

    val read = MessageSerializer.deserializeSchema(
      new ReadChannel(Channels.newChannel(new ByteArrayInputStream(bytes)))
    )
    assertEq(read, schema)
  }
}
