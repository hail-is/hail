package is.hail.types

import is.hail.TestUtils._
import is.hail.collection.FastSeq
import is.hail.expr.ir.IRParser
import is.hail.rvd.{AbstractRVDSpec, RVDType, RVDTypeSerializer}
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils.deepestMessage
import is.hail.variant.{ReferenceGenome, ReferenceGenomeEnv}

import scala.collection.immutable.ArraySeq

import org.json4s.{DefaultFormats, Formats, MappingException}
import org.json4s.jackson.Serialization
import org.junit.jupiter.api.Test

class TypeParsingSuite {

  private val customReference: ReferenceGenome =
    ReferenceGenome("custom", ArraySeq("c", "a", "b"), Map("a" -> 100, "b" -> 2000, "c" -> 30000))

  private val rg = customReference.name

  private val inScope = ReferenceGenomeEnv(Map(rg -> customReference))

  private def typeFormats(rgs: ReferenceGenomeEnv): Formats =
    DefaultFormats +
      new TypeSerializer(rgs) +
      new TStructSerializer(rgs) +
      new TableTypeSerializer(rgs) +
      new MatrixTypeSerializer(rgs) +
      new PTypeSerializer(rgs) +
      new PStructSerializer(rgs) +
      new RVDTypeSerializer(rgs)

  private val rowType = TStruct("l" -> TLocus(rg), "i" -> TInterval(TLocus(rg)), "x" -> TInt32)

  private val tableType = TableType(rowType, FastSeq("l"), TStruct("g" -> TLocus(rg)))

  private val matrixType = MatrixType(
    globalType = TStruct("g" -> TLocus(rg)),
    colKey = FastSeq("s"),
    colType = TStruct("s" -> TString),
    rowKey = FastSeq("l"),
    rowType = rowType,
    entryType = TStruct("e" -> TLocus(rg)),
  )

  private val rowPType = PCanonicalStruct(
    true,
    "l" -> PCanonicalLocus(rg, required = true),
    "c" -> PCanonicalLocus(rg),
    "i" -> PCanonicalInterval(PCanonicalLocus(rg)),
  )

  private val rvdType = RVDType(rowPType, FastSeq("l"))

  @Test def testBuiltinLocusTypesParseWithoutRegistry(): Unit =
    for (name <- ReferenceGenome.hailReferences) {
      assertEq(IRParser.parseType(s"Locus($name)", ReferenceGenomeEnv.empty), TLocus(name))
      assertEq(
        IRParser.parsePType(s"PCLocus($name)", ReferenceGenomeEnv.empty),
        PCanonicalLocus(name),
      )
    }

  @Test def testVirtualTypesRoundTripWithUserDefinedReference(): Unit = {
    assertEq(IRParser.parseType(rowType.parsableString(), inScope), rowType)
    assertEq(IRParser.parseStructType(rowType.parsableString(), inScope), rowType)
    assertEq(IRParser.parseTableType(tableType.toString, inScope), tableType)
    assertEq(IRParser.parseMatrixType(matrixType.toString, inScope), matrixType)
  }

  @Test def testPhysicalTypesRoundTripWithUserDefinedReference(): Unit = {
    assertEq(IRParser.parsePType(rowPType.toString, inScope), rowPType)
    assertEq(IRParser.parseRVDType(rvdType.toString, inScope), rvdType)
  }

  @Test def testSerializersRoundTripWithUserDefinedReference(): Unit = {
    implicit val formats: Formats = typeFormats(inScope)
    assertEq(Serialization.read[Type](Serialization.write[Type](rowType)), rowType)
    assertEq(Serialization.read[TStruct](Serialization.write(rowType)), rowType)
    assertEq(Serialization.read[TableType](Serialization.write(tableType)), tableType)
    assertEq(Serialization.read[MatrixType](Serialization.write(matrixType)), matrixType)
    assertEq(Serialization.read[PStruct](Serialization.write[PStruct](rowPType)), rowPType)
    assertEq(Serialization.read[RVDType](Serialization.write(rvdType)), rvdType)
  }

  @Test def testWriteOnlyFormatsWriteWhatPerRegistryFormatsRead(): Unit = {
    val written = Serialization.write(rvdType)(AbstractRVDSpec.formats)
    assertEq(
      Serialization.read[RVDType](written)(AbstractRVDSpec.formats(inScope), manifest),
      rvdType,
    )
  }

  @Test def testWriteOnlyFormatsRefuseToReadTypes(): Unit = {
    val written = Serialization.write[Type](TInt32)(AbstractRVDSpec.formats)
    val e = intercept[MappingException](
      Serialization.read[Type](written)(AbstractRVDSpec.formats, manifest)
    )
    assertEq(
      deepestMessage(e),
      "IllegalStateException: Cannot read a Type: these json4s Formats only write types. " +
        "Reading a type needs the reference genomes in scope.",
    )
  }

  @Test def testUserDefinedLocusOutOfScopeFails(): Unit = {
    val notInScope = s"Reference genome '$rg' is not in scope"
    interceptFatal(notInScope)(IRParser.parseType(s"Locus($rg)", ReferenceGenomeEnv.empty))
    interceptFatal(notInScope)(IRParser.parsePType(s"PCLocus($rg)", ReferenceGenomeEnv.empty))
    interceptFatal(notInScope)(IRParser.parseTableType(
      tableType.toString,
      ReferenceGenomeEnv.empty,
    ))

    // json4s wraps the failure; the Python error is the deepest cause's message.
    implicit val formats: Formats = typeFormats(ReferenceGenomeEnv.empty)
    val e =
      intercept[MappingException](Serialization.read[Type](Serialization.write[Type](rowType)))
    assertEq(deepestMessage(e), s"HailException: $notInScope.")
  }
}
