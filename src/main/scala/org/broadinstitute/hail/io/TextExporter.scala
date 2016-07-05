package org.broadinstitute.hail.io

import org.apache
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr.Type

trait TextExporter {
  def exportTypes(filename: String, hConf: apache.hadoop.conf.Configuration, info: Array[(String, Type)]) {
    val sb = new StringBuilder
    writeTextFile(filename, hConf) { out =>
      info.foreachBetween { case (name, t) =>
        sb.append(prettyIdentifier(name))
        sb.append(":")
        t.pretty(sb, printAttrs = true, compact = true)
      }(() => sb.append(","))

      out.write(sb.result())
    }
  }
}
