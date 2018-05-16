package is.hail.expr.ir

import is.hail.expr.{BaseIR, MatrixRead, TableImport, TableRead}
import is.hail.utils._

object Pretty {

  def apply(ir: BaseIR): String = {
    val sb = new StringBuilder

    def pretty(ir: BaseIR, depth: Int) {
      sb.append(" " * depth)
      sb += '('
      sb.append(ir.getClass.getName.split("\\.").last)

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
        case _ =>
          val header = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => s""""$x""""
            case Cast(_, typ) => typ.toString
            case NA(typ) => typ.toString
            case Let(name, _, _) => name
            case Ref(name, _) => name
            case ApplyBinaryPrimOp(op, _, _) => op.getClass.getName.split("\\.").last
            case ApplyUnaryPrimOp(op, _) => op.getClass.getName.split("\\.").last
            case GetField(_, name) => name
            case GetTupleElement(_, idx) => idx.toString
            case ArrayMap(_, name, _) => name
            case ArrayFilter(_, name, _) => name
            case ArrayFlatMap(_, name, _) => name
            case ArrayFold(_, _, accumName, valueName, _) => s"$accumName $valueName"
            case ApplyAggOp(_, _, _, aggSig) => aggSig.op.getClass.getName.split("\\.").last
            case SeqOp(_, _, aggSig) => aggSig.op.getClass.getName.split("\\.").last
            case ArrayFor(_, valueName, _) => s"$valueName"
            case ApplyIR(function, _, _) => function
            case Apply(function, _) => function
            case ApplySpecial(function, _) => function
            case In(i, _) => i.toString
            case MatrixRead(typ, partitionCounts, dropCols, dropRows, _) =>
              s"$typ partition_counts=${ partitionCounts.map(_.mkString(",")).getOrElse("None") } ${ if (dropRows) "drop_rows" else "" }${ if (dropCols) "drop_cols" else "" }"
            case TableImport(paths, _, _) =>
              if (paths.length == 1)
                paths.head
              else {
                sb += '\n'
                sb.append(" " * (depth + 2))
                sb.append("(paths\n")
                paths.foreachBetween { p =>
                  sb.append(" " * (depth + 4))
                  StringEscapeUtils.escapeString(p, sb, backticked = false)
                }(sb += '\n')
                sb += ')'

                ""
              }
            case TableRead(path, _, dropRows) =>
              if (dropRows)
                s"${ StringEscapeUtils.escapeString(path) } drop_rows"
              else
                path
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
