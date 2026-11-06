package is.hail.io.columnar

import is.hail.TestUtils._
import is.hail.collection.FastSeq

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
