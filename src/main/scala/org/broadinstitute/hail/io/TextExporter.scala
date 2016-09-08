package org.broadinstitute.hail.io

import org.apache
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.expr.Type

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
