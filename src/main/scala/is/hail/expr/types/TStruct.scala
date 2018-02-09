package is.hail.expr.types

import is.hail.annotations.{Annotation, AnnotationPathException, _}
import is.hail.asm4s.{Code, _}
import is.hail.check.Gen
import is.hail.expr.{EvalContext, HailRep, Parser}
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}

object TStruct {
  private val requiredEmpty = TStruct(Array.empty[Field], true)
  private val optionalEmpty = TStruct(Array.empty[Field], false)

  def empty(required: Boolean = false): TStruct = if (required) requiredEmpty else optionalEmpty

  def apply(required: Boolean, args: (String, Type)*): TStruct =
    TStruct(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => Field(n, t, i) }
      .toArray,
      required)

  def apply(args: (String, Type)*): TStruct =
    apply(false, args: _*)

  def apply(names: java.util.ArrayList[String], types: java.util.ArrayList[Type], required: Boolean): TStruct = {
    val sNames = names.asScala.toArray
    val sTypes = types.asScala.toArray
    if (sNames.length != sTypes.length)
      fatal(s"number of names does not match number of types: found ${ sNames.length } names and ${ sTypes.length } types")

    val t = TStruct(sNames.zip(sTypes): _*)
    t.setRequired(required).asInstanceOf[TStruct]
  }
}

final case class TStruct(fields: IndexedSeq[Field], override val required: Boolean = false) extends Type {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i })

  val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  val fieldNames: Array[String] = fields.map(_.name).toArray

  val fieldRequired: Array[Boolean] = fields.map(_.typ.required).toArray

  val fieldType: Array[Type] = fields.map(_.typ).toArray

  assert(fieldNames.areDistinct(), fieldNames.duplicates())

  override def children: Seq[Type] = fields.map(_.typ)

  def fieldByName(name: String): Field = fields(fieldIdx(name))

  override def canCompare(other: Type): Boolean = other match {
    case t: TStruct => size == t.size && fields.zip(t.fields).forall { case (f1, f2) =>
      f1.name == f2.name && f1.typ.canCompare(f2.typ)
    }
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TStruct(cfields, _) =>
      fields.length == cfields.length &&
        (fields, cfields).zipped.forall { case (f, cf) =>
          f.unify(cf)
        }
    case _ => false
  }

  override def subst() = TStruct(fields.map(f => f.copy(typ = f.typ.subst().asInstanceOf[Type])))


  def index(str: String): Option[Int] = fieldIdx.get(str)

  def selfField(name: String): Option[Field] = fieldIdx.get(name).map(i => fields(i))

  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def field(name: String): Field = fields(fieldIdx(name))

  val size: Int = fields.length

  override def getOption(path: List[String]): Option[Type] =
    if (path.isEmpty)
      Some(this)
    else
      selfField(path.head).map(_.typ).flatMap(t => t.getOption(path.tail))

  override def fieldOption(path: List[String]): Option[Field] =
    if (path.isEmpty)
      None
    else {
      val f = selfField(path.head)
      if (path.length == 1)
        f
      else
        f.flatMap(_.typ.fieldOption(path.tail))
    }

  override def queryTyped(p: List[String]): (Type, Querier) = {
    if (p.isEmpty)
      (this, identity[Annotation])
    else {
      selfField(p.head) match {
        case Some(f) =>
          val (t, q) = f.typ.queryTyped(p.tail)
          val localIndex = f.index
          (t, (a: Any) =>
            if (a == null)
              null
            else
              q(a.asInstanceOf[Row].get(localIndex)))
        case None => throw new AnnotationPathException(s"struct has no field ${ p.head }")
      }
    }
  }

  override def delete(p: List[String]): (Type, Deleter) = {
    if (p.isEmpty)
      (TStruct.empty(), a => null)
    else {
      val key = p.head
      val f = selfField(key) match {
        case Some(f) => f
        case None => throw new AnnotationPathException(s"$key not found")
      }
      val index = f.index
      val (newFieldType, d) = f.typ.delete(p.tail)
      val newType: Type =
        if (newFieldType == TStruct.empty())
          deleteKey(key, f.index)
        else
          updateKey(key, f.index, newFieldType)

      val localDeleteFromRow = newFieldType == TStruct.empty()

      val deleter: Deleter = { a =>
        if (a == null)
          null
        else {
          val r = a.asInstanceOf[Row]

          if (localDeleteFromRow)
            r.deleteField(index)
          else
            r.update(index, d(r.get(index)))
        }
      }
      (newType, deleter)
    }
  }

  def unsafeStructInsert(typeToInsert: Type, path: List[String]): (TStruct, UnsafeInserter) = {
    assert(typeToInsert.isInstanceOf[TStruct] || path.nonEmpty)
    val (t, i) = unsafeInsert(typeToInsert, path)
    (t.asInstanceOf[TStruct], i)
  }

  override def unsafeInsert(typeToInsert: Type, path: List[String]): (Type, UnsafeInserter) = {
    if (path.isEmpty) {
      (typeToInsert, (region, offset, rvb, inserter) => inserter())
    } else {
      val localSize = size
      val key = path.head
      selfField(key) match {
        case Some(f) =>
          val j = f.index
          val (insertedFieldType, fieldInserter) = f.typ.unsafeInsert(typeToInsert, path.tail)

          (updateKey(key, j, insertedFieldType), { (region, offset, rvb, inserter) =>
            rvb.startStruct()
            var i = 0
            while (i < j) {
              if (region != null)
                rvb.addField(this, region, offset, i)
              else
                rvb.setMissing()
              i += 1
            }
            if (region != null && isFieldDefined(region, offset, j))
              fieldInserter(region, loadField(region, offset, j), rvb, inserter)
            else
              fieldInserter(null, 0, rvb, inserter)
            i += 1
            while (i < localSize) {
              if (region != null)
                rvb.addField(this, region, offset, i)
              else
                rvb.setMissing()
              i += 1
            }
            rvb.endStruct()
          })

        case None =>
          val (insertedFieldType, fieldInserter) = TStruct.empty().unsafeInsert(typeToInsert, path.tail)

          (appendKey(key, insertedFieldType), { (region, offset, rvb, inserter) =>
            rvb.startStruct()
            var i = 0
            while (i < localSize) {
              if (region != null)
                rvb.addField(this, region, offset, i)
              else
                rvb.setMissing()
              i += 1
            }
            fieldInserter(null, 0, rvb, inserter)
            rvb.endStruct()
          })
      }
    }
  }

  override def insert(signature: Type, p: List[String]): (Type, Inserter) = {
    if (p.isEmpty)
      (signature, (a, toIns) => toIns)
    else {
      val key = p.head
      val f = selfField(key)
      val keyIndex = f.map(_.index)
      val (newKeyType, keyF) = f
        .map(_.typ)
        .getOrElse(TStruct.empty())
        .insert(signature, p.tail)

      val newSignature = keyIndex match {
        case Some(i) => updateKey(key, i, newKeyType)
        case None => appendKey(key, newKeyType)
      }

      val localSize = fields.size

      val inserter: Inserter = (a, toIns) => {
        val r = if (a == null || localSize == 0) // localsize == 0 catches cases where we overwrite a path
          Row.fromSeq(Array.fill[Any](localSize)(null))
        else
          a.asInstanceOf[Row]
        keyIndex match {
          case Some(i) => r.update(i, keyF(r.get(i), toIns))
          case None => r.append(keyF(null, toIns))
        }
      }
      (newSignature, inserter)
    }
  }

  def structInsert(signature: Type, p: List[String]): (TStruct, Inserter) = {
    require(p.nonEmpty || signature.isInstanceOf[TStruct], s"tried to remap top-level struct to non-struct $signature")
    val (t, f) = insert(signature, p)
    (t.asInstanceOf[TStruct], f)
  }

  def updateKey(key: String, i: Int, sig: Type): Type = {
    assert(fieldIdx.contains(key))

    val newFields = Array.fill[Field](fields.length)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(i) = Field(key, sig, i)
    TStruct(newFields)
  }

  def deleteKey(key: String, index: Int): Type = {
    assert(fieldIdx.contains(key))
    if (fields.length == 1)
      TStruct.empty()
    else {
      val newFields = Array.fill[Field](fields.length - 1)(null)
      for (i <- 0 until index)
        newFields(i) = fields(i)
      for (i <- index + 1 until fields.length)
        newFields(i - 1) = fields(i).copy(index = i - 1)
      TStruct(newFields)
    }
  }

  def appendKey(key: String, sig: Type): TStruct = {
    assert(!fieldIdx.contains(key))
    val newFields = Array.fill[Field](fields.length + 1)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(fields.length) = Field(key, sig, fields.length)
    TStruct(newFields)
  }

  def merge(other: TStruct): (TStruct, Merger) = {
    val intersect = fields.map(_.name).toSet
      .intersect(other.fields.map(_.name).toSet)

    if (intersect.nonEmpty)
      fatal(
        s"""Invalid merge operation: cannot merge structs with same-name ${ plural(intersect.size, "field") }
           |  Found these fields in both structs: [ ${
          intersect.map(s => prettyIdentifier(s)).mkString(", ")
        } ]
           |  Hint: use `drop' or `select' to remove these fields from one side""".stripMargin)

    val newStruct = TStruct(fields ++ other.fields.map(f => f.copy(index = f.index + size)))

    val size1 = size
    val size2 = other.size
    val targetSize = newStruct.size

    val merger = (a1: Annotation, a2: Annotation) => {
      if (a1 == null && a2 == null)
        null
      else {
        val s1 = Option(a1).map(_.asInstanceOf[Row].toSeq)
          .getOrElse(Seq.fill[Any](size1)(null))
        val s2 = Option(a2).map(_.asInstanceOf[Row].toSeq)
          .getOrElse(Seq.fill[Any](size2)(null))
        val newValues = s1 ++ s2
        assert(newValues.size == targetSize)
        Annotation.fromSeq(newValues)
      }
    }

    (newStruct, merger)
  }

  def annotate(other: TStruct): (TStruct, Merger) = {
    val newFieldsBuilder = new ArrayBuilder[(String, Type)]()
    val fieldIdxBuilder = new ArrayBuilder[Int]()
    // In fieldIdxBuilder, positive integers are field indices from the left.
    // Negative integers are the complement of field indices from the right.

    val rightFieldIdx = other.fields.map { f => f.name -> (f.index, f.typ) }.toMap
    val leftFields = fieldNames.toSet

    fields.foreach { f =>
      rightFieldIdx.get(f.name) match {
        case Some((rightIdx, typ)) =>
          fieldIdxBuilder += ~rightIdx
          newFieldsBuilder += f.name -> typ
        case None =>
          fieldIdxBuilder += f.index
          newFieldsBuilder += f.name -> f.typ
      }
    }
    other.fields.foreach { f =>
      if (!leftFields.contains(f.name)) {
        fieldIdxBuilder += ~f.index
        newFieldsBuilder += f.name -> f.typ
      }
    }

    val newStruct = TStruct(newFieldsBuilder.result(): _*)
    val fieldIdx = fieldIdxBuilder.result()
    val leftNulls = Row.fromSeq(Array.fill[Any](size)(null))
    val rightNulls = Row.fromSeq(Array.fill[Any](other.size)(null))

    val annotator = (a1: Annotation, a2: Annotation) => {
      if (a1 == null && a2 == null)
        null
      else {
        val leftValues = if (a1 == null) leftNulls else a1.asInstanceOf[Row]
        val rightValues = if (a2 == null) rightNulls else a2.asInstanceOf[Row]
        val resultValues = new Array[Any](fieldIdx.length)
        var i = 0
        while (i < resultValues.length) {
          val idx = fieldIdx(i)
          if (idx < 0)
            resultValues(i) = rightValues(~idx)
          else
            resultValues(i) = leftValues(idx)
          i += 1
        }
        Row.fromSeq(resultValues)
      }
    }
    newStruct -> annotator
  }

  def filter(set: Set[String], include: Boolean = true): (TStruct, Deleter) = {
    val notFound = set.filter(name => selfField(name).isEmpty).map(prettyIdentifier)
    if (notFound.nonEmpty)
      fatal(
        s"""invalid struct filter operation: ${
          plural(notFound.size, s"field ${ notFound.head }", s"fields [ ${ notFound.mkString(", ") } ]")
        } not found
           |  Existing struct fields: [ ${ fields.map(f => prettyIdentifier(f.name)).mkString(", ") } ]""".stripMargin)

    val fn = (f: Field) =>
      if (include)
        set.contains(f.name)
      else
        !set.contains(f.name)
    filter(fn)
  }

  def ungroup(identifier: String, mangle: Boolean = false): (TStruct, (Row) => Row) = {
    val overlappingNames = mutable.Set[String]()
    val fieldNamesSet = fieldNames.toSet

    val ungroupedFields = fieldOption(identifier) match {
      case None => fatal(s"Struct does not have field with name `$identifier'.")
      case Some(x) => x.typ match {
        case s: TStruct =>
          s.fields.flatMap { fd =>
            (fieldNamesSet.contains(fd.name), mangle) match {
              case (_, true) => Some((identifier + "." + fd.name, fd.typ))
              case (false, false) => Some((fd.name, fd.typ))
              case (true, false) =>
                overlappingNames += fd.name
                None
            }
          }.toArray
        case other => fatal(s"Can only ungroup fields of type Struct, but found type ${ other.toString } for identifier $identifier.")
      }
    }

    if (overlappingNames.nonEmpty)
      fatal(s"Found ${ overlappingNames.size } ${ plural(overlappingNames.size, "ungrouped field name") } overlapping existing struct field names.\n" +
        "Either rename manually or use the 'mangle' option to handle duplicates.\n Overlapping fields:\n  " +
        s"@1", overlappingNames.truncatable("\n  "))

    val fdIndexToUngroup = fieldIdx(identifier)

    val newSignature = TStruct(fields.filterNot(_.index == fdIndexToUngroup).map { fd => (fd.name, fd.typ) } ++ ungroupedFields: _*)

    val origSize = size
    val newSize = newSignature.size

    val ungrouper = (r: Row) => {
      val result = Array.fill[Any](newSize)(null)

      if (r != null) {
        var localIdx = 0

        var i = 0
        while (i < origSize) {
          if (i != fdIndexToUngroup) {
            result(localIdx) = r.get(i)
            localIdx += 1
          }
          i += 1
        }

        val ugr = r.getAs[Row](fdIndexToUngroup)

        if (ugr != null) {
          var j = 0
          while (j < ungroupedFields.length) {
            result(localIdx) = ugr.get(j)
            j += 1
            localIdx += 1
          }
        }
      }

      Row.fromSeq(result)
    }

    (newSignature, ungrouper)
  }

  def group(dest: String, names: Array[String]): (TStruct, (Row) => Row) = {
    val fieldsToGroup = names.zipWithIndex.map { case (name, i) =>
      fieldOption(name) match {
        case None => fatal(s"Struct does not have field with name `$name'.")
        case Some(fd) => fd
      }
    }

    val keepFields = fields.filterNot(fd => names.contains(fd.name) || fd.name == dest)
    val keepIndices = keepFields.map(_.index)

    val groupedTyp = TStruct(fieldsToGroup.map(fd => (fd.name, fd.typ)): _*)
    val finalSignature = TStruct(keepFields.map(fd => (fd.name, fd.typ)) :+ (dest, groupedTyp): _*)

    val newSize = finalSignature.size

    val grouper = (r: Row) => {
      val result = Array.fill[Any](newSize)(null)

      if (r != null) {
        var localIdx = 0
        keepIndices.foreach { i =>
          result(localIdx) = r.get(i)
          localIdx += 1
        }

        assert(localIdx == newSize - 1)
        result(localIdx) = Row.fromSeq(fieldsToGroup.map(fd => r.get(fd.index)))
      }

      Row.fromSeq(result)
    }

    (finalSignature, grouper)
  }

  def parseInStructScope[T >: Null](code: String)(implicit hr: HailRep[T]): (Annotation) => T = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val f = Parser.parseTypedExpr[T](code, ec)

    (a: Annotation) => {
      if (a == null)
        null
      else {
        ec.setAllFromRow(a.asInstanceOf[Row])
        f()
      }
    }
  }

  def ++(that: TStruct): TStruct = {
    val overlapping = fields.map(_.name).toSet.intersect(
      that.fields.map(_.name).toSet)
    if (overlapping.nonEmpty)
      fatal(s"overlapping fields in struct concatenation: ${ overlapping.mkString(", ") }")

    TStruct(fields.map(f => (f.name, f.typ)) ++ that.fields.map(f => (f.name, f.typ)): _*)
  }

  def filter(f: (Field) => Boolean): (TStruct, (Annotation) => Annotation) = {
    val included = fields.map(f)

    val newFields = fields.zip(included)
      .flatMap { case (field, incl) =>
        if (incl)
          Some(field)
        else
          None
      }

    val newSize = newFields.size

    val filterer = (a: Annotation) =>
      if (a == null)
        a
      else if (newSize == 0)
        null
      else {
        val r = a.asInstanceOf[Row]
        val newValues = included.zipWithIndex
          .flatMap {
            case (incl, i) =>
              if (incl)
                Some(r.get(i))
              else None
          }
        assert(newValues.length == newSize)
        Annotation.fromSeq(newValues)
      }

    (TStruct(newFields.zipWithIndex.map { case (f, i) => f.copy(index = i) }), filterer)
  }

  def _toString: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append("Struct{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (size == 0)
        sb.append("Struct { }")
      else {
        sb.append("Struct {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  override def _typeCheck(a: Any): Boolean =
    a.isInstanceOf[Row] && {
      val r = a.asInstanceOf[Row]
      r.length == fields.length &&
        r.toSeq.zip(fields).forall {
          case (v, f) => f.typ.typeCheck(v)
        }
    }

  override def str(a: Annotation): String = JsonMethods.compact(toJSON(a))

  override def genNonmissingValue: Gen[Annotation] = {
    if (size == 0) {
      Gen.const(Annotation.empty)
    } else
      Gen.size.flatMap(fuel =>
        if (size > fuel)
          Gen.uniformSequence(fields.map(f => if (fieldRequired(f.index)) f.typ.genValue else Gen.const(null))).map(a => Annotation(a: _*))
        else
          Gen.uniformSequence(fields.map(f => f.typ.genValue)).map(a => Annotation(a: _*)))
  }

  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double): Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && fields.zip(a1.asInstanceOf[Row].toSeq).zip(a2.asInstanceOf[Row].toSeq)
      .forall {
        case ((f, x1), x2) =>
          f.typ.valuesSimilar(x1, x2, tolerance)
      })

  override def desc: String =
    """
    A ``Struct`` is like a Python tuple where the fields are named and the set of fields is fixed.

    An example of constructing and accessing the fields in a ``Struct`` is

    .. code-block:: text
        :emphasize-lines: 2

        let s = {gene: "ACBD", function: "LOF", nHet: 12} in s.gene
        result: "ACBD"

    A field of the ``Struct`` can also be another ``Struct``. For example, ``va.info.AC`` selects the struct ``info`` from the struct ``va``, and then selects the array ``AC`` from the struct ``info``.
    """

  override def scalaClassTag: ClassTag[Row] = classTag[Row]

  val ordering: ExtendedOrdering =
    ExtendedOrdering.rowOrdering(fields.map(_.typ.ordering).toArray)

  override def unsafeOrdering(missingGreatest: Boolean): UnsafeOrdering = {
    val fieldOrderings = fields.map(_.typ.unsafeOrdering(missingGreatest)).toArray

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        while (i < size) {
          val leftDefined = isFieldDefined(r1, o1, i)
          val rightDefined = isFieldDefined(r2, o2, i)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, loadField(r1, o1, i), r2, loadField(r2, o2, i))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            if (missingGreatest)
              return c
            else
              return -c
          }
          i += 1
        }
        0
      }
    }
  }

  def select(keep: Array[String]): (TStruct, (Row) => Row) = {
    val t = TStruct(keep.map { n =>
      n -> field(n).typ
    }: _*)

    val keepIdx = keep.map(fieldIdx)
    val selectF: Row => Row = { r =>
      Row.fromSeq(keepIdx.map(r.get))
    }
    (t, selectF)
  }

  val (missingIdx, nMissing) = {
    var i = 0
    val a = new Array[Int](size)
    fields.foreach { f =>
      a(f.index) = i
      if (!fieldRequired(f.index))
        i += 1
    }
    (a, i)
  }

  def nMissingBytes: Int = (nMissing + 7) >>> 3

  var byteOffsets: Array[Long] = _
  override val byteSize: Long = {
    val a = new Array[Long](size)

    val bp = new BytePacker()

    var offset: Long = nMissingBytes
    fields.foreach { f =>
      val fSize = f.typ.byteSize
      val fAlignment = f.typ.alignment

      bp.getSpace(fSize, fAlignment) match {
        case Some(start) =>
          a(f.index) = start
        case None =>
          val mod = offset % fAlignment
          if (mod != 0) {
            val shift = fAlignment - mod
            bp.insertSpace(shift, offset)
            offset += (fAlignment - mod)
          }
          a(f.index) = offset
          offset += fSize
      }
    }
    byteOffsets = a
    offset
  }

  override val alignment: Long = {
    if (fields.isEmpty)
      1
    else
      fields.map(_.typ.alignment).max
  }

  override val fundamentalType: TStruct = {
    val fundamentalFieldTypes = fields.map(f => f.typ.fundamentalType)
    if ((fields, fundamentalFieldTypes).zipped
      .forall { case (f, ft) => f.typ == ft })
      this
    else {
      val t = TStruct((fields, fundamentalFieldTypes).zipped.map { case (f, ft) => (f.name, ft) }: _*)
      t.setRequired(required).asInstanceOf[TStruct]
    }
  }

  def allocate(region: Region): Long = {
    region.allocate(alignment, byteSize)
  }

  def clearMissingBits(region: Region, off: Long) {
    var i = 0
    while (i < nMissingBytes) {
      region.storeByte(off + i, 0)
      i += 1
    }
  }

  def clearMissingBits(region: Code[Region], off: Code[Long]): Code[Unit] = {
    var c: Code[Unit] = Code._empty
    var i = 0
    while (i < nMissingBytes) {
      c = Code(c, region.storeByte(off + i.toLong, const(0)))
      i += 1
    }
    c
  }


  def isFieldDefined(rv: RegionValue, fieldIdx: Int): Boolean =
    isFieldDefined(rv.region, rv.offset, fieldIdx)

  def isFieldDefined(region: Region, offset: Long, fieldIdx: Int): Boolean =
    fieldRequired(fieldIdx) || !region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    if (fieldRequired(fieldIdx))
      false
    else
      region.loadBit(offset, missingIdx(fieldIdx))

  def isFieldDefined(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Boolean] =
    !isFieldMissing(region, offset, fieldIdx)

  def setFieldMissing(region: Region, offset: Long, fieldIdx: Int) {
    assert(!fieldRequired(fieldIdx))
    region.setBit(offset, missingIdx(fieldIdx))
  }

  def setFieldMissing(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Unit] = {
    assert(!fieldRequired(fieldIdx))
    region.setBit(offset, missingIdx(fieldIdx))
  }

  def fieldOffset(offset: Long, fieldIdx: Int): Long =
    offset + byteOffsets(fieldIdx)

  def fieldOffset(offset: Code[Long], fieldIdx: Int): Code[Long] =
    offset + byteOffsets(fieldIdx)

  def loadField(rv: RegionValue, fieldIdx: Int): Long = loadField(rv.region, rv.offset, fieldIdx)

  def loadField(region: Region, offset: Long, fieldIdx: Int): Long = {
    val off = fieldOffset(offset, fieldIdx)
    fields(fieldIdx).typ.fundamentalType match {
      case _: TArray | _: TBinary => region.loadAddress(off)
      case _ => off
    }
  }

  def loadField(region: Code[Region], offset: Code[Long], fieldName: String): Code[Long] = {
    val f = field(fieldName)
    loadField(region, fieldOffset(offset, f.index), f.typ)
  }

  def loadField(region: Code[Region], offset: Code[Long], fieldIdx: Int): Code[Long] =
    loadField(region, fieldOffset(offset, fieldIdx), fields(fieldIdx).typ)

  private def loadField(region: Code[Region], fieldOffset: Code[Long], fieldType: Type): Code[Long] = {
    fieldType.fundamentalType match {
      case _: TArray | _: TBinary => region.loadAddress(fieldOffset)
      case _ => fieldOffset
    }
  }

  def uniqueFieldName(base: String): String = {
    val fieldNames = fields.map(_.name).toSet
    if (fieldNames.contains(base)) {
      var i = 0
      var candidate = base + i.toString
      while (fieldNames.contains(candidate)) {
        i += 1
        candidate = base + i.toString
      }
      candidate
    } else
      base
  }
}
