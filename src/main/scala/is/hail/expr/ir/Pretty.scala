package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types.TArray
import is.hail.table.Ascending
import is.hail.utils._
import is.hail.variant.RelationalSpec
import org.json4s.jackson.{JsonMethods, Serialization}

object Pretty {
  def prettyStringLiteral(s: String): String =
    "\"" + StringEscapeUtils.escapeString(s) + "\""

  def prettyStrings(xs: IndexedSeq[String]): String = xs.map(prettyStringLiteral).mkString("(", " ", ")")

  def prettyStringsOpt(x: Option[IndexedSeq[String]]): String = x.map(prettyStrings).getOrElse("None")

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String =
    x.getClass.getName.split("\\.").last

  def prettyAggSignature(aggSig: AggSignature): String = {
    val sb = new StringBuilder
    sb += '('
    sb.append(prettyClass(aggSig.op))
    sb += ' '
    sb.append(aggSig.constructorArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb.append(aggSig.initOpArgs.map(_.map(_.parsableString()).mkString(" (", " ", ")")).getOrElse(" None"))
    sb.append(aggSig.seqOpArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb += ')'
    sb.result()
  }

  def prettyIntOpt(x: Option[Int]): String = x.map(_.toString).getOrElse("None")

  def prettyLongs(x: IndexedSeq[Long]): String = x.mkString("(", " ", ")")

  def prettyInts(x: IndexedSeq[Int]): String = x.mkString("(", " ", ")")

  def prettyLongsOpt(x: Option[IndexedSeq[Long]]): String =
    x.map(prettyLongs).getOrElse("None")

  def prettyIdentifiers(x: IndexedSeq[String]): String = x.map(prettyIdentifier).mkString("(", " ", ")")

  def prettyIdentifiersOpt(x: Option[IndexedSeq[String]]): String = x.map(prettyIdentifiers).getOrElse("None")

  def prettyMatrixReader(reader: MatrixReader): String = {
    import MatrixReader.formats
    s"(MatrixReader ${ prettyStringLiteral(Serialization.write(reader)) })"
  }

  def apply(ir: BaseIR): String = {
    val sb = new StringBuilder

    def prettySeq(xs: Seq[BaseIR], depth: Int) {
      sb.append(" " * depth)
      sb += '('
      xs.foreachBetween(x => pretty(x, depth + 2))(sb += '\n')
      sb += ')'
    }

    def pretty(ir: BaseIR, depth: Int) {
      sb.append(" " * depth)
      sb += '('

      ir match {
        case read: MatrixRead =>
          read.reader match {
            case _: MatrixNativeReader => sb.append("MatrixRead")
            case _: MatrixRangeReader => sb.append("MatrixRange")
          }
        case _ =>
          sb.append(prettyClass(ir) )
      }

      ir match {
        case MakeStruct(fields) =>
          if (fields.nonEmpty) {
            sb += '\n'
            fields.foreachBetween { case (n, a) =>
              sb.append(" " * (depth + 2))
              sb += '('
              sb.append(n)
              sb += '\n'
              pretty(a, depth + 4)
              sb += ')'
            }(sb += '\n')
          }
        case ApplyAggOp(a, ctorArgs, initOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(a, depth + 2)
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
        case ApplyScanOp(a, ctorArgs, initOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(a, depth + 2)
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
        case InitOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case SeqOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case InsertFields(old, fields) =>
          sb += '\n'
          pretty(old, depth + 2)
          if (fields.nonEmpty) {
            sb += '\n'
            fields.foreachBetween { case (n, a) =>
              sb.append(" " * (depth + 2))
              sb += '('
              sb.append(n)
              sb += '\n'
              pretty(a, depth + 4)
              sb += ')'
            }(sb += '\n')
          }
        case TableImport(paths, _, conf) =>
          sb += '\n'
          sb.append(" " * (depth + 2))
          sb.append("(paths\n")
          paths.foreachBetween { p =>
            sb.append(" " * (depth + 4))
            sb.append(prettyStringLiteral(p))
          }(sb += '\n')
          sb += ')'
          sb += '\n'
          sb.append(" " * (depth + 2))
          sb.append("(useCols ")
          conf.useColIndices.foreachBetween(i => sb.append(i))(sb.append(','))
          sb += ')'
          sb += ')'
        case _ =>
          val header = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => prettyStringLiteral(x)
            case Cast(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Let(name, _, _) => prettyIdentifier(name)
            case Ref(name, typ) => s"${ typ.parsableString() } $name"
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => s"(${ prettyClass(op) } ${ op.t1.parsableString() } ${ op.t2.parsableString() })"
            case GetField(_, name) => prettyIdentifier(name)
            case GetTupleElement(_, idx) => idx.toString
            case MakeArray(_, typ) => typ.parsableString()
            case ArrayMap(_, name, _) => prettyIdentifier(name)
            case ArrayFilter(_, name, _) => prettyIdentifier(name)
            case ArrayFlatMap(_, name, _) => prettyIdentifier(name)
            case ArrayFold(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case ArrayFor(_, valueName, _) => prettyIdentifier(valueName)
            case ArraySort(_, _, onKey) => prettyBooleanLiteral(onKey)
            case ApplyIR(function, _, _) => prettyIdentifier(function)
            case Apply(function, _) => prettyIdentifier(function)
            case ApplySpecial(function, _) => prettyIdentifier(function)
            case SelectFields(_, fields) => fields.map(prettyIdentifier).mkString("(", " ", ")")
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"${ typ.parsableString() } $i"
            case Die(message, typ) => typ.parsableString() + " " + prettyStringLiteral(message)
            case Uniroot(name, _, _, _) => prettyIdentifier(name)
            case MatrixRead(typ, _, _, dropCols, dropRows, MatrixNativeReader(path, spec)) =>
              prettyStringLiteral(path) + " " +
                prettyBooleanLiteral(dropCols) + " " +
                prettyBooleanLiteral(dropRows) + " " +
                (if (typ == spec.matrix_type)
                  "None"
                else
                  typ.parsableString())
            case MatrixRead(typ, _, _, dropCols, dropRows, MatrixRangeReader(nRows, nCols, nPartitions)) =>
              s"$nRows $nCols" + " " +
                nPartitions.map(_.toString).getOrElse("None") + " " +
                prettyBooleanLiteral(dropCols) + " " +
                prettyBooleanLiteral(dropRows)
            case MatrixExplodeRows(_, path) => prettyIdentifiers(path)
            case MatrixChooseCols(_, oldIndices) => prettyInts(oldIndices)
            case MatrixMapCols(_, _, newKey) => prettyStringsOpt(newKey)
            case MatrixMapRows(_, _, newKey) =>
              prettyStringsOpt(newKey.map(_._1)) + " " + prettyStringsOpt(newKey.map(_._2))
            case TableImport(paths, _, _) =>
              if (paths.length == 1)
                paths.head
              else {
                sb += '\n'
                sb.append(" " * (depth + 2))
                sb.append("(paths\n")
                paths.foreachBetween { p =>
                  sb.append(" " * (depth + 4))
                  sb.append(prettyStringLiteral(p))
                }(sb += '\n')
                sb += ')'

                ""
              }
            case TableRead(path, spec, typ, dropRows) =>
              prettyStringLiteral(path) + " " +
                prettyBooleanLiteral(dropRows) + " " +
                (if (typ == spec.table_type)
                  "None"
                else
                  typ.parsableString())
            case TableWrite(_, path, overwrite, _) =>
              if (overwrite)
                s"${ StringEscapeUtils.escapeString(path) } overwrite"
              else
                path
            case TableExport(_, path, typesFile, header, exportType) =>
              val args = Array(
                Some(StringEscapeUtils.escapeString(path)),
                Option(typesFile).map(StringEscapeUtils.escapeString(_)),
                if (header) Some("header") else None,
                Some(exportType)
              ).flatten

              sb += '\n'
              args.foreachBetween { a =>
                sb.append(" " * (depth + 2))
                sb.append(a)
              }(sb += '\n')

              ""
            case TableKeyBy(_, keys, nPartitionKeys, sort) =>
              prettyIdentifiers(keys) + " " +
                prettyIntOpt(nPartitionKeys) + " " +
                prettyBooleanLiteral(sort)
            case TableRange(n, nPartitions) => s"$n $nPartitions"
            case TableJoin(_, _, joinType) => joinType
            case TableMapRows(_, _, newKey, preservedKeyFields) =>
              prettyIdentifiersOpt(newKey) + " " + prettyIntOpt(preservedKeyFields)
            case TableExplode(_, field) => field
            case TableParallelize(typ, rows, nPartitions) =>
              val valueType = TArray(typ.rowType)
              typ.parsableString() + " " + valueType.parsableString() + " " +
                prettyStringLiteral(
                  JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(rows, valueType))) + " " +
                prettyIntOpt(nPartitions)
            case TableMapGlobals(_, _, value) =>
              value.t.parsableString() + " " +
                prettyStringLiteral(
                  JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value.value, value.t)))
            case TableOrderBy(_, sortFields) => prettyIdentifiers(sortFields.map(sf =>
              (if (sf.sortOrder == Ascending) "A" else "D") + sf.field))
            case _ => ""
          }

          if (header.nonEmpty) {
            sb += ' '
            sb.append(header)
          }

          val children = ir.children
          if (children.nonEmpty) {
            sb += '\n'
            children.foreachBetween(c => pretty(c, depth + 2))(sb += '\n')
          }
      }

      sb += ')'
    }

    pretty(ir, 0)

    sb.result()
  }
}
