package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.table.Ascending
import is.hail.utils._
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

  def prettySymbols(x: IndexedSeq[Sym]): String = x.mkString("(", " ", ")")

  def prettySymbolsOpt(x: Option[IndexedSeq[Sym]]): String = x.map(prettySymbols).getOrElse("None")

  def apply(ir: BaseIR, elideLiterals: Boolean = false): String = {
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

      sb.append(prettyClass(ir) )

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
        case ApplyAggOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case ApplyScanOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
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
          val header: String = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => prettyStringLiteral(x)
            case Cast(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Literal(typ, value) =>
              s"${ typ.parsableString() } " + (
                  if (!elideLiterals)
                    s"${ prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ))) }"
                  else
                    "<literal value>"
                )
            case Let(name, _, _) => name.toString
            case Ref(name, _) => name.toString
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => prettyClass(op)
            case GetField(_, name) => name.toString
            case GetTupleElement(_, idx) => idx.toString
            case MakeArray(_, typ) => typ.parsableString()
            case ArrayMap(_, name, _) => name.toString
            case ArrayFilter(_, name, _) => name.toString
            case ArrayFlatMap(_, name, _) => name.toString
            case ArrayFold(_, _, accumName, valueName, _) => accumName + " " + valueName
            case ArrayScan(_, _, accumName, valueName, _) => accumName + " " + valueName
            case ArrayFor(_, valueName, _) => valueName.toString
            case AggExplode(_, name, _) => name.toString
            case ArraySort(_, _, onKey) => prettyBooleanLiteral(onKey)
            case ApplyIR(function, _, _) => prettyIdentifier(function)
            case Apply(function, _) => prettyIdentifier(function)
            case ApplySeeded(function, _, seed) => prettyIdentifier(function) + " " + seed.toString
            case ApplySpecial(function, _) => prettyIdentifier(function)
            case SelectFields(_, fields) => fields.mkString("(", " ", ")")
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"${ typ.parsableString() } $i"
            case Die(message, typ) => typ.parsableString()
            case Uniroot(name, _, _, _) => name.toString
            case MatrixRead(typ, dropCols, dropRows, reader) =>
              (if (typ == reader.fullType) "None" else typ.parsableString()) + " " +
              prettyBooleanLiteral(dropCols) + " " +
              prettyBooleanLiteral(dropRows) + " " +
              '"' + StringEscapeUtils.escapeString(Serialization.write(reader)(MatrixReader.formats)) + '"'
            case MatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"'
            case TableToMatrixTable(_, rowKey, colKey, rowFields, colFields, nPartitions) =>
              prettySymbols(rowKey) + " " +
              prettySymbols(colKey) +  " " +
              prettySymbols(rowFields) + " " +
              prettySymbols(colFields) + " " +
              prettyIntOpt(nPartitions)
            case MatrixAnnotateRowsTable(_, _, uid, key) =>
              uid + " " + prettyBooleanLiteral(key.isDefined)
            case MatrixAnnotateColsTable(_, _, uid) =>
              uid.toString
            case MatrixExplodeRows(_, path) => prettySymbols(path)
            case MatrixExplodeCols(_, path) => prettySymbols(path)
            case MatrixRepartition(_, n, strategy) => s"$n $strategy"
            case MatrixChooseCols(_, oldIndices) => prettyInts(oldIndices)
            case MatrixMapCols(_, _, newKey) => prettySymbolsOpt(newKey)
            case MatrixKeyRowsBy(_, keys, isSorted) =>
              prettySymbols(keys) + " " +
                prettyBooleanLiteral(isSorted)
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
            case TableWrite(_, path, overwrite, stageLocally, codecSpecJSONStr) =>
              prettyStringLiteral(path) + " " +
                prettyBooleanLiteral(overwrite) + " " +
                prettyBooleanLiteral(stageLocally) + " " +
                (if (codecSpecJSONStr == null)
                  "None"
                else
                  prettyStringLiteral(codecSpecJSONStr))
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
            case TableKeyBy(_, keys, isSorted) =>
              prettySymbols(keys) + " " +
                prettyBooleanLiteral(isSorted)
            case TableRange(n, nPartitions) => s"$n $nPartitions"
            case TableRepartition(_, n, strategy) => s"$n $strategy"
            case TableHead(_, n) => n.toString
            case TableJoin(_, _, joinType, joinKey) => s"$joinType $joinKey"
            case TableLeftJoinRightDistinct(_, _, root) => root.toString
            case TableIntervalJoin(_, _, root) => root.toString
            case TableMultiWayZipJoin(_, dataName, globalName) =>
              s"$dataName $globalName"
            case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
              prettyIntOpt(nPartitions) + " " + bufferSize.toString
            case TableExplode(_, path) => prettySymbols(path)
            case TableParallelize(_, nPartitions) =>
                prettyIntOpt(nPartitions)
            case TableOrderBy(_, sortFields) => sortFields.map(sf =>
              s"${ if (sf.sortOrder == Ascending) "A" else "D" } ${ sf.field }").mkString("(", " ", ")")
            case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
              s"$entriesFieldName $colsFieldName"
            case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
              s"$entriesFieldName $colsFieldName" +
                prettySymbols(colKey)
            case MatrixToMatrixApply(_, config) => prettyStringLiteral(Serialization.write(config)(RelationalFunctions.formats))
            case MatrixToTableApply(_, config) => prettyStringLiteral(Serialization.write(config)(RelationalFunctions.formats))
            case TableToTableApply(_, config) => prettyStringLiteral(Serialization.write(config)(RelationalFunctions.formats))
            case TableRename(_, rowMap, globalMap) =>
              val rowKV = rowMap.toArray
              val globalKV = globalMap.toArray
              s"${ prettySymbols(rowKV.map(_._1)) } ${ prettySymbols(rowKV.map(_._2)) } " +
                s"${ prettySymbols(globalKV.map(_._1)) } ${ prettySymbols(globalKV.map(_._2)) }"
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
