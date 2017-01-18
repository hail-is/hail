package is.hail.io

import org.apache
import is.hail.utils._
import is.hail.expr.Type

trait TextExporter {
  def exportTypes(filename: String, hConf: apache.hadoop.conf.Configuration, info: Array[(String, Type)]) {
    val sb = new StringBuilder
    hConf.writeTextFile(filename) { out =>
      info.foreachBetween { case (name, t) =>
        sb.append(prettyIdentifier(name))
        sb.append(":")
        t.pretty(sb, printAttrs = true, compact = true)
      } { sb += ',' }

      out.write(sb.result())
    }
  }
}
