package is.hail.types.arrow

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.collection.FastSeq
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.SRNGState
import is.hail.variant.ReferenceGenome

import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters._

import java.nio.ByteBuffer

import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{ArrowType, DictionaryEncoding, Field, FieldType, Schema}
import org.junit.jupiter.api.Test

class ArrowSchemaConversionSuite {

  private val small =
    ReferenceGenome("small", FastSeq("1", "2", "X"), Map("1" -> 10, "2" -> 10, "X" -> 10))

  private val rgs: Map[String, ReferenceGenome] =
    ReferenceGenome.builtinReferences() + (small.name -> small)

  private def toField(t: PType, name: String = "x"): Field =
    ArrowSchemaConversion.toArrowField(name, t, rgs)

  private def fromField(f: Field): PType =
    ArrowSchemaConversion.fromArrowField(f, rgs)

  private def leaf(name: String, nullable: Boolean, t: ArrowType): Field =
    new Field(name, new FieldType(nullable, t, null), FastSeq.empty[Field].asJava)

  private def node(name: String, nullable: Boolean, t: ArrowType, children: Field*): Field =
    new Field(name, new FieldType(nullable, t, null), children.asJava)

  private def children(f: Field): IndexedSeq[Field] = f.getChildren.asScala.toIndexedSeq

  def roundTripTypes() = ArraySeq[PType](
    PBoolean(true),
    PBoolean(false),
    PInt32(true),
    PInt32(false),
    PInt64(true),
    PInt64(false),
    PFloat32(true),
    PFloat64(false),
    PCanonicalString(true),
    PCanonicalString(false),
    PCanonicalBinary(true),
    PCanonicalBinary(false),
    PCanonicalCall(true),
    PCanonicalCall(false),
    PCanonicalLocus(ReferenceGenome.GRCh37, true),
    PCanonicalLocus(small.name, false),
    PCanonicalArray(PInt32(true), false),
    PCanonicalArray(PCanonicalArray(PCanonicalString(false), true), true),
    PCanonicalSet(PInt32(false), true),
    PCanonicalSet(PCanonicalLocus(small.name, true), false),
    PCanonicalDict(PCanonicalString(true), PInt32(false), false),
    PCanonicalDict(PCanonicalStruct(true, "a" -> PInt32(true)), PCanonicalCall(true), true),
    PCanonicalInterval(PInt32(false), false),
    PCanonicalInterval(PCanonicalLocus(small.name, true), true),
    PCanonicalNDArray(PFloat64(true), 2, false),
    PCanonicalNDArray(PInt32(true), 0, true),
    PCanonicalTuple(true, PInt32(false), PCanonicalString(true)),
    PCanonicalTuple(
      FastSeq(PTupleField(1, PInt32(true)), PTupleField(3, PCanonicalString(false))),
      false,
    ),
    PCanonicalStruct(false),
    PCanonicalStruct(
      true,
      "locus" -> PCanonicalLocus(small.name, true),
      "alleles" -> PCanonicalArray(PCanonicalString(true), false),
      "info" -> PCanonicalStruct(
        false,
        "AC" -> PCanonicalArray(PInt32(false), false),
        "filters" -> PCanonicalSet(PCanonicalString(true), false),
      ),
      "entries" -> PCanonicalArray(
        PCanonicalStruct(
          true,
          "GT" -> PCanonicalCall(false),
          "PL" -> PCanonicalNDArray(PInt32(true), 1),
        ),
        true,
      ),
    ),
  )

  @Test def testRoundTripTypesDataProvider(): Unit = roundTripTypes(): Unit

  @ParameterizedTest("roundTripTypes")
  def testRoundTrip(t: PType): Unit =
    assertEq(fromField(toField(t)), t)

  @ParameterizedTest("roundTripTypes")
  def testRoundTripThroughIpcSchema(t: PType): Unit = {
    val schema = new Schema(FastSeq(toField(t)).asJava)
    val bytes = schema.serializeAsMessage()
    val read = Schema.deserializeMessage(ByteBuffer.wrap(bytes))
    assertEq(read, schema)
    assertEq(fromField(read.getFields.get(0)), t)
  }

  @Test def testSchemaRoundTrip(): Unit = {
    val t = PCanonicalStruct(
      true,
      "a" -> PInt32(true),
      "b" -> PCanonicalArray(PCanonicalString(false), false),
    )
    val schema = ArrowSchemaConversion.toArrowSchema(t, rgs)
    assertEq(schema.getFields.asScala.map(_.getName).toSeq, Seq("a", "b"))
    assertEq(ArrowSchemaConversion.fromArrowSchema(schema, rgs), t)
  }

  @Test def testPrimitives(): Unit = {
    assertEq(toField(PBoolean(true)), leaf("x", false, ArrowType.Bool.INSTANCE))
    assertEq(toField(PInt32(false)), leaf("x", true, new ArrowType.Int(32, true)))
    assertEq(toField(PInt64(true)), leaf("x", false, new ArrowType.Int(64, true)))
    assertEq(
      toField(PFloat32(true)),
      leaf("x", false, new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
    )
    assertEq(
      toField(PFloat64(true)),
      leaf("x", false, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
    )
    assertEq(toField(PCanonicalString(false)), leaf("x", true, ArrowType.Utf8.INSTANCE))
    assertEq(toField(PCanonicalBinary(false)), leaf("x", true, ArrowType.Binary.INSTANCE))
  }

  @Test def testArray(): Unit =
    assertEq(
      toField(PCanonicalArray(PInt32(true), false)),
      node(
        "x",
        true,
        ArrowType.List.INSTANCE,
        leaf("element", false, new ArrowType.Int(32, true)),
      ),
    )

  @Test def testSet(): Unit = {
    val f = toField(PCanonicalSet(PInt32(false), true))
    assertEq(
      f,
      node(
        "x",
        false,
        SetExtensionType,
        leaf("element", true, new ArrowType.Int(32, true)),
      ),
    )
    assertEq(SetExtensionType.storageType(), ArrowType.List.INSTANCE)
  }

  @Test def testDict(): Unit =
    assertEq(
      toField(PCanonicalDict(PCanonicalString(true), PInt32(false), false)),
      node(
        "x",
        true,
        new ArrowType.Map(true),
        node(
          "entries",
          false,
          ArrowType.Struct.INSTANCE,
          leaf("key", false, ArrowType.Utf8.INSTANCE),
          leaf("value", true, new ArrowType.Int(32, true)),
        ),
      ),
    )

  @Test def testDictKeysAreAlwaysRequired(): Unit = {
    val t = PCanonicalDict(PCanonicalString(false), PInt32(false), false)
    val f = toField(t)
    assert(!children(children(f).head).head.isNullable)
    assertEq(fromField(f), PCanonicalDict(PCanonicalString(true), PInt32(false), false))
  }

  @Test def testStruct(): Unit =
    assertEq(
      toField(PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PCanonicalString(false))),
      node(
        "x",
        true,
        ArrowType.Struct.INSTANCE,
        leaf("a", false, new ArrowType.Int(32, true)),
        leaf("b", true, ArrowType.Utf8.INSTANCE),
      ),
    )

  @Test def testTuple(): Unit =
    assertEq(
      toField(
        PCanonicalTuple(
          FastSeq(PTupleField(0, PInt32(true)), PTupleField(2, PCanonicalString(false))),
          true,
        )
      ),
      node(
        "x",
        false,
        TupleExtensionType,
        leaf("0", false, new ArrowType.Int(32, true)),
        leaf("2", true, ArrowType.Utf8.INSTANCE),
      ),
    )

  @Test def testCall(): Unit = {
    val f = toField(PCanonicalCall(false))
    assertEq(f, leaf("x", true, CallExtensionType.Current))
    assertEq(CallExtensionType.Current.storageType(), new ArrowType.Int(32, true))
    assertEq(CallExtensionType.Current.serialize(), """{"encoding":1}""")
  }

  @Test def testLocus(): Unit = {
    val f = toField(PCanonicalLocus(small.name, true))
    val ext = new LocusExtensionType(small.name, small.contigs)
    assertEq(f, leaf("x", false, ext))
    assertEq(ext.storageType(), new ArrowType.Int(64, true))
    assertEq(ext.serialize(), """{"reference_genome":"small","contigs":["1","2","X"]}""")
  }

  @Test def testInterval(): Unit = {
    val locus = new LocusExtensionType(small.name, small.contigs)
    assertEq(
      toField(PCanonicalInterval(PCanonicalLocus(small.name, true), false)),
      node(
        "x",
        true,
        IntervalExtensionType,
        leaf("start", false, locus),
        leaf("end", false, locus),
        leaf("includes_start", false, ArrowType.Bool.INSTANCE),
        leaf("includes_end", false, ArrowType.Bool.INSTANCE),
      ),
    )
  }

  @Test def testNDArray(): Unit =
    assertEq(
      toField(PCanonicalNDArray(PFloat64(true), 3, false)),
      node(
        "x",
        true,
        VariableShapeTensorExtensionType,
        node(
          "data",
          false,
          ArrowType.List.INSTANCE,
          leaf("element", false, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
        ),
        node(
          "shape",
          false,
          new ArrowType.FixedSizeList(3),
          leaf("element", false, new ArrowType.Int(32, true)),
        ),
      ),
    )

  @Test def testExtensionNames(): Unit = {
    assertEq(CallExtensionType.Current.extensionName(), "hail.call")
    assertEq(new LocusExtensionType(small.name, small.contigs).extensionName(), "hail.locus")
    assertEq(SetExtensionType.extensionName(), "hail.set")
    assertEq(IntervalExtensionType.extensionName(), "hail.interval")
    assertEq(TupleExtensionType.extensionName(), "hail.tuple")
    assertEq(VariableShapeTensorExtensionType.extensionName(), "arrow.variable_shape_tensor")
  }

  @Test def testUnstorableTypes(): Unit = {
    interceptFatal("cannot be stored")(toField(PVoid))
    interceptFatal("cannot be stored")(toField(StoredSTypePType(SRNGState(None), true)))
    interceptFatal("cannot be stored")(
      toField(PCanonicalStruct(false, "rng" -> StoredSTypePType(SRNGState(None), true)))
    )
  }

  @Test def testUnknownReferenceGenomeOnWrite(): Unit =
    interceptFatal("reference genome 'nope'")(toField(PCanonicalLocus("nope", true)))

  @Test def testUnknownReferenceGenomeOnRead(): Unit =
    interceptFatal("reference genome 'nope'")(
      fromField(leaf("x", false, new LocusExtensionType("nope", FastSeq("1"))))
    )

  @Test def testContigMismatchOnRead(): Unit =
    interceptFatal("contigs")(
      fromField(leaf("x", false, new LocusExtensionType(small.name, FastSeq("1", "X", "2"))))
    )

  @Test def testUnknownCallEncoding(): Unit =
    interceptFatal("call encoding 2")(fromField(leaf("x", false, new CallExtensionType(2))))

  def unsupportedArrowFields() = ArraySeq[Field](
    leaf("x", true, ArrowType.LargeUtf8.INSTANCE),
    leaf("x", true, ArrowType.LargeBinary.INSTANCE),
    leaf("x", true, new ArrowType.Int(8, true)),
    leaf("x", true, new ArrowType.Int(32, false)),
    leaf("x", true, new ArrowType.FloatingPoint(FloatingPointPrecision.HALF)),
    node("x", true, ArrowType.LargeList.INSTANCE, leaf("element", true, ArrowType.Bool.INSTANCE)),
    node(
      "x",
      true,
      new ArrowType.Map(false),
      node(
        "entries",
        false,
        ArrowType.Struct.INSTANCE,
        leaf("key", false, ArrowType.Utf8.INSTANCE),
        leaf("value", true, ArrowType.Utf8.INSTANCE),
      ),
    ),
    node(
      "x",
      true,
      new ArrowType.Map(true),
      node(
        "entries",
        false,
        ArrowType.Struct.INSTANCE,
        leaf("key", true, ArrowType.Utf8.INSTANCE),
        leaf("value", true, ArrowType.Utf8.INSTANCE),
      ),
    ),
    node(
      "x",
      true,
      TupleExtensionType,
      leaf("a", true, ArrowType.Utf8.INSTANCE),
    ),
    node(
      "x",
      true,
      IntervalExtensionType,
      leaf("start", true, ArrowType.Utf8.INSTANCE),
      leaf("end", true, new ArrowType.Int(32, true)),
      leaf("includes_start", false, ArrowType.Bool.INSTANCE),
      leaf("includes_end", false, ArrowType.Bool.INSTANCE),
    ),
    node(
      "x",
      true,
      VariableShapeTensorExtensionType,
      node(
        "data",
        false,
        ArrowType.List.INSTANCE,
        leaf("element", true, new ArrowType.Int(32, true)),
      ),
      node(
        "shape",
        false,
        new ArrowType.FixedSizeList(1),
        leaf("element", false, new ArrowType.Int(32, true)),
      ),
    ),
    new Field(
      "x",
      new FieldType(
        true,
        new ArrowType.Int(64, true),
        null,
        Map(
          ArrowType.ExtensionType.EXTENSION_METADATA_KEY_NAME -> "hail.unknown",
          ArrowType.ExtensionType.EXTENSION_METADATA_KEY_METADATA -> "{}",
        ).asJava,
      ),
      FastSeq.empty[Field].asJava,
    ),
    new Field(
      "x",
      new FieldType(
        true,
        ArrowType.Utf8.INSTANCE,
        new DictionaryEncoding(0, false, new ArrowType.Int(32, true)),
      ),
      FastSeq.empty[Field].asJava,
    ),
    node(
      "x",
      true,
      ArrowType.Struct.INSTANCE,
      leaf("a", true, ArrowType.Utf8.INSTANCE),
      leaf("a", true, ArrowType.Bool.INSTANCE),
    ),
  )

  @Test def testUnsupportedArrowFieldsDataProvider(): Unit = unsupportedArrowFields(): Unit

  @ParameterizedTest("unsupportedArrowFields")
  def testUnsupportedArrowField(f: Field): Unit =
    interceptFatal("columnar format")(fromField(f))
}
