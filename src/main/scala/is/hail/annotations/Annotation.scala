package is.hail.annotations

import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils.{ArrayBuilder, Interval}
import is.hail.variant._
import is.hail.utils._
import org.apache.spark.sql.Row

object Annotation {

  final val COL_HEAD = "sa"

  final val ROW_HEAD = "va"

  final val GLOBAL_HEAD = "global"

  final val ENTRY_HEAD = "g"

  val empty: Annotation = Row()

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = Array.fill[Annotation](n)(Annotation.empty)

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case null => "Null"
      case r: Row =>
        "Struct:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${ printAnnotation(elem, nSpace + 4) }"""
          }
            .mkString("\n")
      case a => a.toString + ": " + a.getClass.getSimpleName
    }
  }

  def expandType(t: Type): Type = t match {
    case tc: ComplexType => expandType(tc.representation)
    case TArray(elementType, req) =>
      TArray(expandType(elementType), req)
    case TStruct(fields, req) =>
      TStruct(fields.map { f => f.copy(typ = expandType(f.typ)) }, req)
    case TTuple(types, req) =>
      TTuple(types.map { t => expandType(t) }, req)
    case TSet(elementType, req) =>
      TArray(expandType(elementType), req)
    case TDict(keyType, valueType, req) =>
      TArray(TStruct(
        "key" -> expandType(keyType),
        "value" -> expandType(valueType)), req)
    case _ => t
  }

  def expandAnnotation(a: Annotation, t: Type): Annotation =
    if (a == null)
      null
    else
      t match {
        case _: TLocus => a.asInstanceOf[Locus].toRow

        case TArray(elementType, _) =>
          a.asInstanceOf[IndexedSeq[_]].map(expandAnnotation(_, elementType))

        case t: TBaseStruct =>
          Row.fromSeq((a.asInstanceOf[Row].toSeq, t.types).zipped.map { case (ai, typ) =>
            expandAnnotation(ai, typ)
          })

        case TSet(elementType, _) =>
          (a.asInstanceOf[Set[_]]
            .toArray[Any]: IndexedSeq[_])
            .map(expandAnnotation(_, elementType))

        case TDict(keyType, valueType, _) =>
          (a.asInstanceOf[Map[String, _]]

            .toArray[(Any, Any)]: IndexedSeq[(Any, Any)])
            .map { case (k, v) => Annotation(expandAnnotation(k, keyType), expandAnnotation(v, valueType)) }

        case TInterval(pointType, _) =>
          val i = a.asInstanceOf[Interval]
          Annotation(
            expandAnnotation(i.start, pointType),
            expandAnnotation(i.end, pointType))

        // including TChar, TSample
        case _ => a
      }

  def flattenType(t: Type): Type = t match {
    case t: TStruct =>
      val flatFields = t.fields.flatMap { f =>
        flattenType(f.typ) match {
          case t2: TStruct =>
            t2.fields.map { f2 => (f.name + "." + f2.name, f2.typ) }

          case _ => Seq(f.name -> f.typ)
        }
      }

      TStruct(flatFields: _*)

    case _ => t
  }

  def flattenAnnotation(a: Annotation, t: Type): Annotation = t match {
    case t: TStruct =>
      val s =
        if (a == null)
          Seq.fill(t.fields.length)(null)
        else
          a.asInstanceOf[Row].toSeq

      val fs = (s, t.fields).zipped.flatMap { case (ai, f) =>
        f.typ match {
          case t: TStruct =>
            flattenAnnotation(ai, f.typ).asInstanceOf[Row].toSeq

          case _ =>
            Seq(ai)
        }
      }
      Row.fromSeq(fs)

    case _ => a
  }

  def apply(args: Any*): Annotation = Row.fromSeq(args)

  def fromSeq(values: Seq[Any]): Annotation = Row.fromSeq(values)

  def buildInserter(code: String, t: TStruct, ec: EvalContext, expectedHead: String): (TStruct, Inserter) = {
    val (paths, types, f) = Parser.parseAnnotationExprs(code, ec, Some(expectedHead))

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val finalType = (paths, types).zipped.foldLeft(t) { case (t, (ids, signature)) =>
      val (s, i) = t.structInsert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val insF = (left: Annotation, right: Annotation) => {
      ec.setAll(left, right)

      var newAnnotation = left
      val queries = f()
      queries.indices.foreach { i =>
        newAnnotation = inserters(i)(newAnnotation, queries(i))
      }
      newAnnotation
    }

    (finalType, insF)
  }

  def copy(t: Type, a: Annotation): Annotation = {
    if (a == null)
      return null

    t match {
      case t: TBaseStruct =>
        val r = a.asInstanceOf[Row]
        Row(Array.tabulate(r.size)(i => Annotation.copy(t.types(i), r(i))): _*)

      case t: TArray =>
        a.asInstanceOf[IndexedSeq[Annotation]].map(Annotation.copy(t.elementType, _))

      case t: TSet =>
        a.asInstanceOf[Set[Annotation]].map(Annotation.copy(t.elementType, _))

      case t: TDict =>
        a.asInstanceOf[Map[Annotation, Annotation]]
          .map { case (k, v) => (Annotation.copy(t.keyType, k), Annotation.copy(t.valueType, v)) }

      case t: TInterval =>
        val i = a.asInstanceOf[Interval]
        i.copy(start = Annotation.copy(t.pointType, i.start), end = Annotation.copy(t.pointType, i.end))

      case _ => a
    }
  }

  def safeFromRegionValue(t: Type, rv: RegionValue): Annotation =
    safeFromRegionValue(t, rv.region, rv.offset)

  def safeFromRegionValue(t: Type, region: Region, offset: Long): Annotation = t match {
    case _: TBoolean =>
      region.loadBoolean(offset)
    case _: TInt32 | _: TCall => region.loadInt(offset)
    case _: TInt64 => region.loadLong(offset)
    case _: TFloat32 => region.loadFloat(offset)
    case _: TFloat64 => region.loadDouble(offset)
    case t: TArray =>
      safeFromArrayRegionValue(t, region, offset)
    case t: TSet =>
      safeFromArrayRegionValue(t, region, offset).toSet
    case _: TString => UnsafeRow.readString(region, offset)
    case _: TBinary => UnsafeRow.readBinary(region, offset)
    case td: TDict =>
      val a = safeFromArrayRegionValue(td, region, offset)
      a.map(r => r.asInstanceOf[Row]).map(r => (r.get(0), r.get(1))).toMap
    case t: TBaseStruct =>
      safeFromBaseStructRegionValue(t, region, offset)
    case t: TLocus =>
      UnsafeRow.readLocus(region, offset, t.rg)
    case t: TInterval =>
      val ft = t.fundamentalType.asInstanceOf[TStruct]
      val start: Annotation =
        if (ft.isFieldDefined(region, offset, 0))
          safeFromRegionValue(t.pointType, region, ft.loadField(region, offset, 0))
        else
          null
      val end =
        if (ft.isFieldDefined(region, offset, 1))
          safeFromRegionValue(t.pointType, region, ft.loadField(region, offset, 1))
        else
          null
      val includesStart = safeFromRegionValue(
        TBooleanRequired,
        region,
        ft.loadField(region, offset, 2)).asInstanceOf[Boolean]
      val includesEnd = safeFromRegionValue(
        TBooleanRequired,
        region,
        ft.loadField(region, offset, 3)).asInstanceOf[Boolean]
      Interval(start, end, includesStart, includesEnd)
  }

  def safeFromArrayRegionValue(
    t: TContainer,
    region: Region,
    offset: Long
  ): IndexedSeq[Annotation] = {
    val length = t.loadLength(region, offset)
    val a = new Array[Annotation](length)
    var i = 0
    while (i < length) {
      a(i) =
        safeFromRegionValue(
          t.elementType,
          region,
          t.loadElement(region, offset, length, i))
      i += 1
    }
    a.toFastIndexedSeq
  }

  def safeFromBaseStructRegionValue(
    t: TBaseStruct,
    region: Region,
    offset: Long
  ): Row =
    Row(t.fields.map(f =>
      safeFromRegionValue(
        f.typ,
        region,
        t.loadField(region, offset, f.index))
    ).toArray: _*)
}
