package org.broadinstitute.hail.vcf

// ##FORMAT=<ID=GT, ...> is a format field
// GT:AD:... is a format
// 1/0:5,2:... is a genotype

case class FormatField(id: String,
  fields: Map[String, String])

case class Header(lines: Array[String],
  sampleIds: Array[String],
  formatFields: Map[String, FormatField]) {
}

object Header {
  def apply(lines: Array[String]): Header = {
    val headerLine = lines.find(line => line.length >= 2 && line(0) == '#' && line(1) != '#').get

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val b = new StringBuilder()

    val formatPrefix = "##FORMAT=<"
    val formatSuffix = ">"

    val formatFields = lines.flatMap(line => {
      if (line.startsWith("##FORMAT")) {
        // FIXME error checking
        val bit = line.iterator.buffered

        def collectWhile(p: (Char) => Boolean): String = {
          b.clear()
          while (p(bit.head))
            b += bit.next()
          b.result()
        }

        def parseValue(): String = {
          if (bit.head == '"') {
            bit.next()
            b.clear()
            while (bit.head != '"')
              b += bit.next()
            bit.next() // "
            b.result()
          } else
            collectWhile(_ != ',')
        }

        def parseBody(): Map[String, String] = {
          if (bit.head == '>')
            Map()
          else {
            if (bit.head == ',')
              bit.next() // ,
            val key = collectWhile(_ != '=')
            bit.next() // =
            val value = parseValue()
            parseBody().updated(key, value)
          }
        }

        def parsePrefix() = {
          while (bit.head != '<')
            bit.next()
          bit.next() // '<'
        }

        def parseFields(): Map[String, String] = {
          parsePrefix()
          parseBody()
        }

        val fields = parseFields()
        Some(FormatField(fields("ID"), fields))
      } else
        None
    })
      .map(format => (format.id, format))
      .toMap

    new Header(lines, sampleIds, formatFields)
  }
}
