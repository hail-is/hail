package is.hail.types.arrow

import is.hail.types.physical._
import is.hail.utils.fatal
import is.hail.variant.ReferenceGenome

import scala.jdk.CollectionConverters._

import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}

/** Conversion between Hail physical types and the Arrow schema of the columnar format.
  *
  * Arrow nullability carries Hail requiredness at every level, with one exception: dict keys are
  * always non-nullable, so a dict whose key type is optional reads back with a required key type.
  *
  * Arrow -> Hail accepts exactly the Arrow types that Hail -> Arrow produces.
  */
object ArrowSchemaConversion {
  ExtensionTypes.register()

  val ElementFieldName: String = "element"
  val EntriesFieldName: String = "entries"
  val KeyFieldName: String = "key"
  val ValueFieldName: String = "value"

  private val Int32 = new ArrowType.Int(32, true)
  private val Int64 = new ArrowType.Int(64, true)
  private val Float32 = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
  private val Float64 = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)

  private val IntervalFieldNames =
    IndexedSeq("start", "end", "includes_start", "includes_end")

  private val TensorFieldNames = IndexedSeq("data", "shape")

  def toArrowSchema(t: PStruct, referenceGenomes: Map[String, ReferenceGenome]): Schema =
    new Schema(t.fields.map(f => toArrowField(f.name, f.typ, referenceGenomes)).asJava)

  def toArrowField(name: String, t: PType, referenceGenomes: Map[String, ReferenceGenome])
    : Field = {
    def field(arrowType: ArrowType, children: Field*): Field =
      mkField(name, !t.required, arrowType, children: _*)

    def child(name: String, t: PType): Field =
      toArrowField(name, t, referenceGenomes)

    t match {
      case _: PBoolean => field(ArrowType.Bool.INSTANCE)
      case _: PInt32 => field(Int32)
      case _: PInt64 => field(Int64)
      case _: PFloat32 => field(Float32)
      case _: PFloat64 => field(Float64)
      case _: PString => field(ArrowType.Utf8.INSTANCE)
      case _: PBinary => field(ArrowType.Binary.INSTANCE)
      case _: PCall => field(CallExtensionType.Current)
      case t: PLocus =>
        val rg = referenceGenomes.getOrElse(
          t.rg,
          fatal(s"field '$name': reference genome '${t.rg}' is not defined"),
        )
        field(new LocusExtensionType(rg.name, rg.contigs))
      case t: PInterval =>
        field(
          IntervalExtensionType,
          child(IntervalFieldNames(0), t.pointType),
          child(IntervalFieldNames(1), t.pointType),
          mkField(IntervalFieldNames(2), nullable = false, ArrowType.Bool.INSTANCE),
          mkField(IntervalFieldNames(3), nullable = false, ArrowType.Bool.INSTANCE),
        )
      case t: PNDArray =>
        field(
          VariableShapeTensorExtensionType,
          mkField(
            TensorFieldNames(0),
            nullable = false,
            ArrowType.List.INSTANCE,
            child(ElementFieldName, t.elementType),
          ),
          mkField(
            TensorFieldNames(1),
            nullable = false,
            new ArrowType.FixedSizeList(t.nDims),
            mkField(ElementFieldName, nullable = false, Int32),
          ),
        )
      case t: PSet => field(SetExtensionType, child(ElementFieldName, t.elementType))
      case t: PDict =>
        field(
          new ArrowType.Map( /*keysSorted=*/ true),
          mkField(
            EntriesFieldName,
            nullable = false,
            ArrowType.Struct.INSTANCE,
            child(KeyFieldName, t.keyType.setRequired(true)),
            child(ValueFieldName, t.valueType),
          ),
        )
      case t: PArray => field(ArrowType.List.INSTANCE, child(ElementFieldName, t.elementType))
      case t: PTuple => field(TupleExtensionType, t.fields.map(f => child(f.name, f.typ)): _*)
      case t: PStruct =>
        field(ArrowType.Struct.INSTANCE, t.fields.map(f => child(f.name, f.typ)): _*)
      case _ =>
        fatal(s"field '$name': Hail type ${t.virtualType} cannot be stored in the columnar format")
    }
  }

  def fromArrowSchema(schema: Schema, referenceGenomes: Map[String, ReferenceGenome])
    : PCanonicalStruct =
    PCanonicalStruct(
      structFields(schema.getFields.asScala.toIndexedSeq, referenceGenomes),
      required = true,
    )

  def fromArrowField(field: Field, referenceGenomes: Map[String, ReferenceGenome]): PType = {
    val name = field.getName
    val required = !field.isNullable
    val children = field.getChildren.asScala.toIndexedSeq

    def unsupported(reason: String): Nothing =
      fatal(
        s"field '$name' of Arrow type ${field.getType} is not part of the columnar format: $reason"
      )

    def checkChildren(expected: IndexedSeq[String]): Unit =
      if (children.map(_.getName) != expected)
        unsupported(
          s"expected child fields ${expected.mkString("[", ", ", "]")}, found ${children.map(_.getName).mkString("[", ", ", "]")}"
        )

    def onlyChild(): Field =
      if (children.length == 1) children.head
      else unsupported(s"expected 1 child field, found ${children.length}")

    def convert(f: Field): PType = fromArrowField(f, referenceGenomes)

    def requiredChild(f: Field): Field =
      if (f.isNullable) unsupported(s"child field '${f.getName}' must be non-nullable")
      else f

    if (field.getDictionary != null)
      unsupported("dictionary encoding is not supported")

    field.getType match {
      case _: CallExtensionType if field.getType == CallExtensionType.Current =>
        PCanonicalCall(required)
      case t: CallExtensionType =>
        unsupported(s"unknown call encoding ${t.encoding}")
      case t: LocusExtensionType =>
        val rg = referenceGenomes.getOrElse(
          t.referenceGenome,
          fatal(s"field '$name': reference genome '${t.referenceGenome}' is not defined"),
        )
        if (rg.contigs != t.contigs)
          fatal(
            s"field '$name': the stored contigs of reference genome '${t.referenceGenome}' " +
              "differ from the contigs of the defined reference genome"
          )
        PCanonicalLocus(rg.name, required)
      case SetExtensionType =>
        PCanonicalSet(convert(onlyChild()), required)
      case IntervalExtensionType =>
        checkChildren(IntervalFieldNames)
        val IndexedSeq(start, end, includesStart, includesEnd) = children
        val pointType = convert(start)
        if (convert(end) != pointType)
          unsupported("interval 'start' and 'end' have different types")
        for (f <- Seq(includesStart, includesEnd))
          if (f.isNullable || f.getType != ArrowType.Bool.INSTANCE || !f.getChildren.isEmpty)
            unsupported(s"interval '${f.getName}' must be a non-nullable bool")
        PCanonicalInterval(pointType, required)
      case TupleExtensionType =>
        val fields = children.map { c =>
          val idx = c.getName.toIntOption.filter(i => i >= 0 && i.toString == c.getName).getOrElse(
            unsupported(s"tuple field name '${c.getName}' is not a non-negative integer")
          )
          PTupleField(idx, convert(c))
        }
        if (fields.map(_.index).distinct.length != fields.length)
          unsupported("tuple field names are not distinct")
        PCanonicalTuple(fields, required)
      case VariableShapeTensorExtensionType =>
        checkChildren(TensorFieldNames)
        val IndexedSeq(data, shape) = children
        val nDims = shape.getType match {
          case t: ArrowType.FixedSizeList => t.getListSize
          case t => unsupported(s"tensor 'shape' must be a fixed_size_list, found $t")
        }
        requiredChild(shape)
        if (shape.getChildren.size != 1 || convert(shape.getChildren.get(0)) != PInt32(true))
          unsupported("tensor 'shape' must contain non-nullable int32")
        if (requiredChild(data).getType != ArrowType.List.INSTANCE || data.getChildren.size != 1)
          unsupported("tensor 'data' must be a list")
        val elementType = convert(requiredChild(data.getChildren.get(0)))
        if (elementType.containsPointers)
          unsupported(s"tensor element type ${elementType.virtualType} is not supported")
        PCanonicalNDArray(elementType, nDims, required)
      case t: ArrowType.ExtensionType =>
        unsupported(s"unknown extension type '${t.extensionName}'")
      case _
          if field.getMetadata.containsKey(ArrowType.ExtensionType.EXTENSION_METADATA_KEY_NAME) =>
        unsupported(
          s"unregistered extension type '${field.getMetadata.get(ArrowType.ExtensionType.EXTENSION_METADATA_KEY_NAME)}'"
        )
      case ArrowType.Bool.INSTANCE => PBoolean(required)
      case Int32 => PInt32(required)
      case Int64 => PInt64(required)
      case Float32 => PFloat32(required)
      case Float64 => PFloat64(required)
      case ArrowType.Utf8.INSTANCE => PCanonicalString(required)
      case ArrowType.Binary.INSTANCE => PCanonicalBinary(required)
      case ArrowType.List.INSTANCE => PCanonicalArray(convert(onlyChild()), required)
      case t: ArrowType.Map =>
        if (!t.getKeysSorted) unsupported("map keys must be sorted")
        val entries = requiredChild(onlyChild())
        if (entries.getType != ArrowType.Struct.INSTANCE || entries.getChildren.size != 2)
          unsupported("map entries must be a struct with two fields")
        val key = requiredChild(entries.getChildren.get(0))
        PCanonicalDict(convert(key), convert(entries.getChildren.get(1)), required)
      case ArrowType.Struct.INSTANCE =>
        PCanonicalStruct(structFields(children, referenceGenomes), required)
      case _ => unsupported("unsupported type")
    }
  }

  private def structFields(
    fields: IndexedSeq[Field],
    referenceGenomes: Map[String, ReferenceGenome],
  ): IndexedSeq[PField] = {
    val duplicates = fields.map(_.getName).groupBy(identity).collect {
      case (name, names) if names.length > 1 => name
    }
    if (duplicates.nonEmpty)
      fatal(
        s"duplicate field names are not part of the columnar format: ${duplicates.mkString(", ")}"
      )
    fields.zipWithIndex.map { case (f, i) =>
      PField(f.getName, fromArrowField(f, referenceGenomes), i)
    }
  }

  private def mkField(name: String, nullable: Boolean, arrowType: ArrowType, children: Field*)
    : Field =
    new Field(name, new FieldType(nullable, arrowType, /*dictionary=*/ null), children.asJava)
}
