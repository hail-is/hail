package is.hail.io.columnar

import is.hail.collection.FastSeq
import is.hail.types.arrow.ArrowSchemaConversion
import is.hail.types.physical.PStruct
import is.hail.types.virtual._
import is.hail.utils.fatal
import is.hail.variant.ReferenceGenome

import scala.jdk.CollectionConverters._

import java.io.ByteArrayOutputStream
import java.nio.channels.Channels
import java.util.Base64

import org.apache.arrow.vector.ipc.WriteChannel
import org.apache.arrow.vector.ipc.message.{IpcOption, MessageSerializer}
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.parquet.schema.{LogicalTypeAnnotation, MessageType, Type => ParquetType, Types}
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName
import org.apache.parquet.schema.Type.Repetition

/** Schema-level rules of the columnar format that apply to every data file (spec section 4.6). */
object ColumnarSchema {

  /** The Arrow field metadata key holding a field's Parquet field ID. */
  val FieldIdKey: String = "PARQUET:field_id"

  /** Fails if any field of `t`, at any depth, holds RNG state, or an ndarray whose elements aren't
    * numeric or bool: the Hail types that can't be stored (spec sections 5 and 5.4).
    */
  def checkSupported(what: String, t: TStruct): Unit = {
    def check(path: String, t: Type): Unit = t match {
      case TRNGState => fatal(s"$what field '$path' has type $t, which can't be stored")
      case t: TNDArray if !isTensorElement(t.elementType) =>
        fatal(
          s"$what field '$path' has type $t, which can't be stored: an ndarray's elements must " +
            "be numeric or bool"
        )
      case t => components(t).foreach { case (suffix, c) => check(path + suffix, c) }
    }
    t.fields.foreach(f => check(f.name, f.typ))
  }

  /** Whether `t` can be an element of Arrow's tensor extension types, whose readers require a
    * fixed-width value type with no extension type: a struct or an extension's storage won't do.
    */
  private def isTensorElement(t: Type): Boolean = t match {
    case TInt32 | TInt64 | TFloat32 | TFloat64 | TBoolean => true
    case _ => false
  }

  /** The types `t` is directly made of, each with the suffix that extends a field's path to it. A
    * dict is a container of key-value structs.
    */
  private def components(t: Type): IndexedSeq[(String, Type)] = t match {
    case t: TContainer => FastSeq("[]" -> t.elementType)
    case t: TNDArray => FastSeq("[]" -> t.elementType)
    case t: TInterval => FastSeq(".start" -> t.pointType)
    case t: TBaseStruct => t.fields.map(f => s".${f.name}" -> f.typ)
    case _ => FastSeq()
  }

  /** The reference genomes of every locus in `types`, at any depth, that aren't builtin, sorted by
    * name (spec section 3). Fails on a reference genome that has a builtin's name but isn't that
    * builtin, since the manifest refers to builtins by name.
    */
  def nonBuiltinReferenceGenomes(types: Type*): IndexedSeq[ReferenceGenome] = {
    def referenceGenomes(t: Type): Iterator[ReferenceGenome] = t match {
      case t: TLocus => Iterator(t.rg)
      case t => components(t).iterator.flatMap { case (_, c) => referenceGenomes(c) }
    }
    val builtins = ReferenceGenome.builtinReferences()
    val nonBuiltin = types.iterator.flatMap(referenceGenomes)
      .filterNot(rg => builtins.get(rg.name).contains(rg))
      .distinct
      .toIndexedSeq
      .sortBy(_.name)
    nonBuiltin.find(rg => builtins.contains(rg.name)).foreach { rg =>
      fatal(
        s"columnar writer: reference genome '${rg.name}' has the name of a builtin reference " +
          "genome but a different definition, so it can't be stored"
      )
    }
    nonBuiltin
  }

  /** `default`, or `default` with the smallest numeric suffix that doesn't collide with `taken`. */
  def internalFieldName(default: String, taken: Iterable[String]): String = {
    val names = taken.toSet
    Iterator.from(0).map(i => if (i == 0) default else s"$default$i").find(!names(_)).get
  }

  /** The Arrow schema of a data file holding the fields of `t`, preceded by a non-null `int64` row
    * ID field named `rowIdField` if given, with field IDs assigned.
    */
  def fileSchema(t: PStruct, rowIdField: Option[String]): Schema = {
    val rowId = rowIdField.map { name =>
      new Field(
        name,
        new FieldType(false, new ArrowType.Int(64, true), null),
        FastSeq.empty[Field].asJava,
      )
    }
    val fields = ArrowSchemaConversion.toArrowSchema(t).getFields.asScala
    withFieldIds(new Schema((rowId ++ fields).asJava))
  }

  /** Assigns field IDs depth-first (pre-order) from 1 to every field except a map's entries field,
    * recording each as [[FieldIdKey]] metadata. Existing field metadata is kept.
    */
  def withFieldIds(schema: Schema): Schema = {
    var next = 0

    def assign(f: Field, isMapEntries: Boolean): Field = {
      val metadata = new java.util.LinkedHashMap[String, String]()
      if (f.getMetadata != null) metadata.putAll(f.getMetadata)
      if (!isMapEntries) {
        next += 1
        metadata.put(FieldIdKey, next.toString)
      }
      val isMap = f.getType.isInstanceOf[ArrowType.Map]
      val children = f.getChildren.asScala.map(assign(_, isMapEntries = isMap))
      new Field(
        f.getName,
        new FieldType(f.isNullable, f.getType, f.getDictionary, metadata),
        children.asJava,
      )
    }

    new Schema(
      schema.getFields.asScala.map(assign(_, isMapEntries = false)).asJava,
      schema.getCustomMetadata,
    )
  }

  /** `schema` as an encapsulated Arrow IPC Schema message (continuation marker, little-endian
    * metadata length, flatbuffer padded to 8 bytes, no body), base64-encoded per RFC 4648 section
    * 4. This is the encoding of the manifest's `schema` fields and of the `ARROW:schema` footer
    * key.
    */
  def serialize(schema: Schema): String = {
    val out = new ByteArrayOutputStream()
    MessageSerializer.serialize(
      new WriteChannel(Channels.newChannel(out)),
      schema,
      IpcOption.DEFAULT,
    )
    Base64.getEncoder.encodeToString(out.toByteArray)
  }

  /** The name of the Parquet message (root) type. */
  val ParquetMessageName: String = "schema"

  /** The name of the repeated group of the standard Parquet LIST encoding. */
  val ParquetListName: String = "list"

  /** The name of the repeated group of the standard Parquet MAP encoding. */
  val ParquetMapEntriesName: String = "key_value"

  /** The Parquet translation of `schema`, using the standard three-level LIST and standard MAP
    * encodings. A `fixed_size_list` is a LIST, and an extension type is its storage type. Each node
    * carries the field ID recorded in its Arrow field's [[FieldIdKey]] metadata, if any.
    */
  def toParquet(schema: Schema): MessageType =
    new MessageType(ParquetMessageName, schema.getFields.asScala.map(toParquetType).asJava)

  private def toParquetType(f: Field): ParquetType = {
    val repetition = if (f.isNullable) Repetition.OPTIONAL else Repetition.REQUIRED
    val id = Option(f.getMetadata).flatMap(m => Option(m.get(FieldIdKey))).map(_.toInt)
    val children = f.getChildren.asScala

    def primitive(name: PrimitiveTypeName, annotation: LogicalTypeAnnotation = null) = {
      val b = Types.primitive(name, repetition).as(annotation)
      id.foreach(b.id)
      b.named(f.getName)
    }

    def group(annotation: LogicalTypeAnnotation, fields: Seq[ParquetType]) = {
      val b = Types.buildGroup(repetition).as(annotation).addFields(fields: _*)
      id.foreach(b.id)
      b.named(f.getName)
    }

    def list(element: Field) = {
      val list = Types.repeatedGroup().addField(toParquetType(element)).named(ParquetListName)
      group(LogicalTypeAnnotation.listType(), FastSeq(list))
    }

    val storageType = f.getType match {
      case t: ArrowType.ExtensionType => t.storageType()
      case t => t
    }

    storageType match {
      case ArrowType.Bool.INSTANCE => primitive(PrimitiveTypeName.BOOLEAN)
      case t: ArrowType.Int if t.getIsSigned && t.getBitWidth == 32 =>
        primitive(PrimitiveTypeName.INT32)
      case t: ArrowType.Int if t.getIsSigned && t.getBitWidth == 64 =>
        primitive(PrimitiveTypeName.INT64)
      case t: ArrowType.FloatingPoint if t.getPrecision == FloatingPointPrecision.SINGLE =>
        primitive(PrimitiveTypeName.FLOAT)
      case t: ArrowType.FloatingPoint if t.getPrecision == FloatingPointPrecision.DOUBLE =>
        primitive(PrimitiveTypeName.DOUBLE)
      case ArrowType.Utf8.INSTANCE =>
        primitive(PrimitiveTypeName.BINARY, LogicalTypeAnnotation.stringType())
      case ArrowType.Binary.INSTANCE => primitive(PrimitiveTypeName.BINARY)
      case ArrowType.List.INSTANCE | _: ArrowType.FixedSizeList => list(children.head)
      case _: ArrowType.Map =>
        val entries = children.head.getChildren.asScala.map(toParquetType)
        val keyValue = Types.repeatedGroup().addFields(entries.toSeq: _*).named(
          ParquetMapEntriesName
        )
        group(LogicalTypeAnnotation.mapType(), FastSeq(keyValue))
      case ArrowType.Struct.INSTANCE => group(null, children.map(toParquetType).toSeq)
      case t =>
        fatal(s"field '${f.getName}': Arrow type $t is not yet supported by the columnar writer")
    }
  }
}
